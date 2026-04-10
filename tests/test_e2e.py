"""End-to-end tests for llamactl -- exercises the CLI as a black box with real file I/O."""

import httpx
import json
import llamactl
import pytest
from llamactl import app
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch


def _parse_models_ini(path):
    """Parse models.ini which has a bare 'version = 1' before any section header."""
    import configparser

    text = path.read_text()
    # Strip the bare key=value lines before the first section header
    lines = text.splitlines(keepends=True)
    filtered = []
    seen_section = False
    for line in lines:
        if line.strip().startswith("["):
            seen_section = True
        if seen_section:
            filtered.append(line)
    ini = configparser.ConfigParser()
    ini.read_string("".join(filtered))
    return ini

runner = CliRunner()


class TestSwitchThenVerify:
    """Full switch workflow: invoke CLI -> read back models.ini -> validate contents."""

    @pytest.fixture()
    def _env(self, tmp_path, presets_file, models_ini, mock_subprocess_run):
        """Set up all three source types with real files on disk."""
        # HF model symlink
        hf_model = models_ini.parent / "test-model.gguf"
        hf_model.write_bytes(b"\x00" * 128)

        # Local model
        local_model = tmp_path / "local-model.gguf"
        local_model.write_bytes(b"\x00" * 64)

        from tests.conftest import SAMPLE_PRESETS

        patched = {**SAMPLE_PRESETS, "test-local": {**SAMPLE_PRESETS["test-local"], "path": str(local_model)}}
        presets_file.write_text(json.dumps(patched))

        self.hf_model = hf_model
        self.local_model = local_model
        self.models_ini = models_ini
        yield

    @pytest.mark.e2e
    def test_switch_hf_produces_parseable_ini(self, _env):
        result = runner.invoke(app, ["switch", "test-hf"])
        assert result.exit_code == 0

        ini = _parse_models_ini(self.models_ini)
        assert set(ini.sections()) == {"*", "test-hf"}
        assert ini["*"]["c"] == "16384"
        assert ini["*"]["n-gpu-layers"] == "99"
        assert ini["test-hf"]["model"] == str(self.hf_model)
        assert ini["test-hf"]["jinja"] == "true"

    @pytest.mark.e2e
    def test_switch_local_no_jinja_in_ini(self, _env):
        result = runner.invoke(app, ["switch", "test-local"])
        assert result.exit_code == 0

        ini = _parse_models_ini(self.models_ini)
        assert "test-local" in ini.sections()
        assert "jinja" not in ini["test-local"]
        assert ini["test-local"]["c"] == "8192"

    @pytest.mark.e2e
    def test_sequential_switches_overwrite_ini(self, _env, ollama_tree):
        """Switching presets overwrites the previous models.ini entirely."""
        from tests.conftest import SAMPLE_PRESETS

        # First switch to HF
        runner.invoke(app, ["switch", "test-hf"])
        ini1 = self.models_ini.read_text()
        assert "[test-hf]" in ini1

        # Now switch to local
        result = runner.invoke(app, ["switch", "test-local"])
        assert result.exit_code == 0
        ini2 = self.models_ini.read_text()
        assert "[test-local]" in ini2
        assert "[test-hf]" not in ini2


class TestSwitchErrorCascades:
    """Error paths produce actionable messages without leaving partial state."""

    @pytest.mark.e2e
    def test_unknown_preset_lists_available(self, presets_file):
        result = runner.invoke(app, ["switch", "nonexistent"])
        assert result.exit_code == 1
        assert "Unknown preset" in result.output
        assert "test-hf" in result.output
        assert "test-ollama" in result.output

    @pytest.mark.e2e
    def test_missing_hf_model_suggests_symlink(self, presets_file, models_ini):
        result = runner.invoke(app, ["switch", "test-hf"])
        assert result.exit_code == 1
        assert "not found" in result.output
        assert "ln -s" in result.output

    @pytest.mark.e2e
    def test_missing_ollama_model_suggests_pull(self, presets_file, models_ini):
        result = runner.invoke(app, ["switch", "test-ollama"])
        assert result.exit_code == 1
        assert "not found" in result.output
        assert "lctl pull" in result.output

    @pytest.mark.e2e
    def test_missing_local_model(self, presets_file, models_ini):
        result = runner.invoke(app, ["switch", "test-local"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @pytest.mark.e2e
    def test_failed_switch_does_not_write_ini(self, presets_file, models_ini):
        """A failed switch must not leave a partial models.ini behind."""
        assert not models_ini.exists()
        runner.invoke(app, ["switch", "test-hf"])
        assert not models_ini.exists()


class TestPresetsE2E:
    """presets command reads real presets.json and displays all entries."""

    @pytest.mark.e2e
    def test_all_presets_listed(self, presets_file):
        result = runner.invoke(app, ["presets"])
        assert result.exit_code == 0
        for name in ("test-hf", "test-ollama", "test-local"):
            assert name in result.output

    @pytest.mark.e2e
    def test_presets_show_source_type(self, presets_file):
        result = runner.invoke(app, ["presets"])
        assert "hf" in result.output
        assert "ollama" in result.output
        assert "local" in result.output

    @pytest.mark.e2e
    def test_presets_show_detail(self, presets_file):
        result = runner.invoke(app, ["presets"])
        assert "test-org/test-model-GGUF:Q4_K_M" in result.output
        assert "test-model:7b" in result.output
        assert "/tmp/test-model.gguf" in result.output

    @pytest.mark.e2e
    def test_presets_missing_file(self, tmp_path):
        with patch.object(llamactl, "PRESETS_PATH", tmp_path / "nope.json"):
            result = runner.invoke(app, ["presets"])
        assert result.exit_code == 1


class TestNiahE2E:
    """niah end-to-end: CLI flags -> prompt construction -> HTTP -> output."""

    @pytest.fixture(autouse=True)
    def _niah_env(self, tmp_path):
        ini = tmp_path / "models.ini"
        ini.write_text("version = 1\n\n[*]\nc = 32768\n\n[test-model]\nmodel = /tmp/test.gguf\nc = 32768\n")
        with patch.object(llamactl, "MODELS_INI_PATH", ini):
            yield

    def _mock_chat_response(self, content):
        resp = MagicMock()
        resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        resp.raise_for_status = MagicMock()
        return resp

    @pytest.mark.e2e
    def test_pass_output_format(self):
        with patch.object(llamactl.httpx, "post", return_value=self._mock_chat_response(
            "Eat a sandwich at Dolores Park on a sunny day."
        )):
            result = runner.invoke(app, ["niah"])

        assert result.exit_code == 0
        assert "NIAH test:" in result.output
        assert "model=test-model" in result.output
        assert "depth=50%" in result.output
        assert "Answer:" in result.output
        assert "Result: PASS" in result.output

    @pytest.mark.e2e
    def test_fail_output_format(self):
        with patch.object(llamactl.httpx, "post", return_value=self._mock_chat_response(
            "I have no idea."
        )):
            result = runner.invoke(app, ["niah"])

        assert result.exit_code == 0
        assert "Result: FAIL" in result.output

    @pytest.mark.e2e
    def test_depth_flag_reflected_in_output(self):
        with patch.object(llamactl.httpx, "post", return_value=self._mock_chat_response(
            "sandwich dolores park sunny"
        )):
            result = runner.invoke(app, ["niah", "--depth", "75"])

        assert "depth=75%" in result.output

    @pytest.mark.e2e
    def test_server_down_error(self):
        with patch.object(llamactl.httpx, "post", side_effect=httpx.ConnectError("refused")):
            result = runner.invoke(app, ["niah"])

        assert result.exit_code == 1
        assert "not responding" in result.output

    @pytest.mark.e2e
    def test_http_error_shows_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_resp
        )
        with patch.object(llamactl.httpx, "post", return_value=mock_resp):
            result = runner.invoke(app, ["niah"])

        assert result.exit_code == 1
        assert "500" in result.output


class TestStartStopRestartE2E:
    """start/stop/restart commands produce expected output."""

    @pytest.mark.e2e
    def test_start_output(self, mock_immortalctl):
        mock_immortalctl.return_value.stdout = ""
        mock_immortalctl.return_value.stderr = ""
        result = runner.invoke(app, ["start"])
        assert result.exit_code == 0
        assert "Started" in result.output

    @pytest.mark.e2e
    def test_stop_output(self, mock_immortalctl):
        mock_immortalctl.return_value.stdout = ""
        mock_immortalctl.return_value.stderr = ""
        result = runner.invoke(app, ["stop"])
        assert result.exit_code == 0
        assert "Stopped" in result.output

    @pytest.mark.e2e
    def test_restart_output(self, mock_subprocess_run):
        result = runner.invoke(app, ["restart"])
        assert result.exit_code == 0
        assert "Restarted" in result.output
        assert "auto-recovers" in result.output


class TestHelpE2E:
    """Invoking with no subcommand shows help."""

    @pytest.mark.e2e
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "status" in result.output
        assert "switch" in result.output
        assert "pull" in result.output
