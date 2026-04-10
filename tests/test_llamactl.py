"""Unit tests for llamactl."""

import httpx
import json
import llamactl
import os
import pytest
from llamactl import app
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch

runner = CliRunner()


# -- _needs_sudo ---------------------------------------------------------


class TestNeedsSudo:
    @pytest.mark.unit
    def test_returns_true_when_socket_missing(self, tmp_path):
        with patch.object(llamactl, "IMMORTAL_SOCKET", tmp_path / "no_such_sock"):
            assert llamactl._needs_sudo() is True

    @pytest.mark.unit
    def test_returns_false_when_socket_readable(self, tmp_path):
        sock = tmp_path / "immortal.sock"
        sock.touch()
        with patch.object(llamactl, "IMMORTAL_SOCKET", sock):
            assert llamactl._needs_sudo() is False

    @pytest.mark.unit
    def test_returns_true_when_socket_not_readable(self, tmp_path):
        sock = tmp_path / "immortal.sock"
        sock.touch()
        sock.chmod(0o000)
        try:
            with patch.object(llamactl, "IMMORTAL_SOCKET", sock):
                if os.getuid() != 0:
                    assert llamactl._needs_sudo() is True
        finally:
            sock.chmod(0o644)


# -- _load_presets --------------------------------------------------------


class TestLoadPresets:
    @pytest.mark.unit
    def test_loads_valid_presets(self, presets_file):
        result = llamactl._load_presets()
        assert "test-hf" in result
        assert result["test-ollama"]["source"] == "ollama"

    @pytest.mark.unit
    def test_exits_when_file_missing(self, tmp_path):
        from click.exceptions import Exit

        with patch.object(llamactl, "PRESETS_PATH", tmp_path / "nope.json"):
            with pytest.raises(Exit):
                llamactl._load_presets()


# -- _resolve_ollama_model_path -------------------------------------------


class TestResolveOllamaModelPath:
    @pytest.mark.unit
    def test_resolves_model_with_tag(self, ollama_tree):
        result = llamactl._resolve_ollama_model_path("test-model:7b")
        assert result is not None
        assert result.exists()
        assert result.name == "sha256-abc123def456"

    @pytest.mark.unit
    def test_resolves_model_default_tag(self, ollama_tree):
        # Create a "latest" manifest
        manifest_dir = ollama_tree / "manifests" / "registry.ollama.ai" / "library" / "test-model"
        (manifest_dir / "latest").write_text(
            json.dumps(
                {
                    "layers": [
                        {
                            "mediaType": "application/vnd.ollama.image.model",
                            "digest": "sha256:abc123def456",
                        }
                    ]
                }
            )
        )
        result = llamactl._resolve_ollama_model_path("test-model")
        assert result is not None
        assert result.exists()

    @pytest.mark.unit
    def test_returns_none_for_missing_model(self, ollama_tree):
        result = llamactl._resolve_ollama_model_path("nonexistent:latest")
        assert result is None

    @pytest.mark.unit
    def test_returns_none_when_no_model_layer(self, ollama_tree):
        manifest_dir = ollama_tree / "manifests" / "registry.ollama.ai" / "library" / "empty-model"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "latest").write_text(
            json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.license", "digest": "sha256:lic"}]})
        )
        result = llamactl._resolve_ollama_model_path("empty-model:latest")
        assert result is None


# -- Templates ------------------------------------------------------------


class TestModelsIniTemplate:
    @pytest.mark.unit
    def test_renders_basic_fields(self):
        result = llamactl.MODELS_INI_TEMPLATE.render(
            name="mymodel",
            source="hf",
            context=16384,
            jinja=False,
            hf_repo="org/repo",
            model_path="/tmp/model.gguf",
        )
        assert "[mymodel]" in result
        assert "model = /tmp/model.gguf" in result
        assert "c = 16384" in result
        assert "jinja" not in result

    @pytest.mark.unit
    def test_renders_jinja_flag(self):
        result = llamactl.MODELS_INI_TEMPLATE.render(
            name="jmodel",
            source="hf",
            context=8192,
            jinja=True,
            hf_repo="",
            model_path="/tmp/j.gguf",
        )
        assert "jinja = true" in result


class TestOpencodeTemplate:
    @pytest.mark.unit
    def test_renders_provider_config(self):
        result = llamactl.OPENCODE_TEMPLATE.render(
            server_url="http://127.0.0.1:8080",
            model_id="test",
            display_name="Test Model",
            context=32768,
        )
        parsed = json.loads(result)
        provider = parsed["provider"]["llama.cpp"]
        assert provider["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
        assert "test" in provider["models"]
        assert provider["models"]["test"]["limit"]["context"] == 32768


# -- CLI commands ---------------------------------------------------------


class TestStatusCommand:
    @pytest.mark.unit
    def test_status_shows_running_info(self, mock_immortalctl, mock_httpx_get):
        mock_immortalctl.return_value.stdout = "llama-server: up (pid 1234)"
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "up" in result.output
        assert "model-a" in result.output

    @pytest.mark.unit
    def test_status_not_running(self, mock_immortalctl):
        mock_immortalctl.return_value.stdout = ""
        mock_immortalctl.return_value.returncode = 1
        result = runner.invoke(app, ["status"])
        assert "not running" in result.output

    @pytest.mark.unit
    def test_status_server_not_responding(self, mock_immortalctl):
        mock_immortalctl.return_value.stdout = "llama-server: up (pid 1234)"
        with patch.object(llamactl.httpx, "get", side_effect=httpx.ConnectError("refused")):
            result = runner.invoke(app, ["status"])
            assert "not responding" in result.output


class TestStartCommand:
    @pytest.mark.unit
    def test_start(self, mock_immortalctl):
        mock_immortalctl.return_value.stdout = "Started"
        result = runner.invoke(app, ["start"])
        assert result.exit_code == 0
        mock_immortalctl.assert_called_once_with("start", "llama-server")


class TestStopCommand:
    @pytest.mark.unit
    def test_stop(self, mock_immortalctl):
        mock_immortalctl.return_value.stdout = "Stopped"
        result = runner.invoke(app, ["stop"])
        assert result.exit_code == 0
        mock_immortalctl.assert_called_once_with("stop", "llama-server")


class TestRestartCommand:
    @pytest.mark.unit
    def test_restart_kills_and_reports(self, mock_subprocess_run):
        result = runner.invoke(app, ["restart"])
        assert result.exit_code == 0
        assert "Restarted" in result.output
        mock_subprocess_run.assert_called_once_with(["pkill", "-f", "llama-server"], capture_output=True)


class TestModelsCommand:
    @pytest.mark.unit
    def test_lists_models(self, mock_httpx_get):
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
        assert "model-a" in result.output
        assert "model-b" in result.output

    @pytest.mark.unit
    def test_models_server_down(self):
        with patch.object(llamactl.httpx, "get", side_effect=httpx.ConnectError("refused")):
            result = runner.invoke(app, ["models"])
            assert result.exit_code == 1


class TestPresetsCommand:
    @pytest.mark.unit
    def test_lists_presets(self, presets_file):
        result = runner.invoke(app, ["presets"])
        assert result.exit_code == 0
        assert "test-hf" in result.output
        assert "test-ollama" in result.output
        assert "test-local" in result.output


class TestSwitchCommand:
    @pytest.mark.unit
    def test_switch_unknown_preset(self, presets_file):
        result = runner.invoke(app, ["switch", "nonexistent"])
        assert result.exit_code == 1
        assert "Unknown preset" in result.output

    @pytest.mark.unit
    def test_switch_hf_preset(self, presets_file, models_ini, mock_subprocess_run):
        model_file = models_ini.parent / "test-model.gguf"
        model_file.touch()
        with patch.object(llamactl, "MODELS_INI_PATH", models_ini):
            result = runner.invoke(app, ["switch", "test-hf"])
        assert result.exit_code == 0
        assert "test-hf" in result.output
        assert models_ini.exists()
        content = models_ini.read_text()
        assert "[test-hf]" in content
        assert "c = 16384" in content

    @pytest.mark.unit
    def test_switch_hf_missing_model_file(self, presets_file, models_ini):
        # model_file symlink doesn't exist
        result = runner.invoke(app, ["switch", "test-hf"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @pytest.mark.unit
    def test_switch_ollama_preset(self, presets_file, models_ini, ollama_tree, mock_subprocess_run):
        # Point the preset at the ollama_tree model
        from tests.conftest import SAMPLE_PRESETS

        patched = {**SAMPLE_PRESETS, "test-ollama": {**SAMPLE_PRESETS["test-ollama"], "ollama_model": "test-model:7b"}}
        presets_file.write_text(json.dumps(patched))

        result = runner.invoke(app, ["switch", "test-ollama"])
        assert result.exit_code == 0
        assert "test-ollama" in result.output

    @pytest.mark.unit
    def test_switch_ollama_missing(self, presets_file, models_ini):
        result = runner.invoke(app, ["switch", "test-ollama"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @pytest.mark.unit
    def test_switch_local_preset(self, presets_file, models_ini, tmp_path, mock_subprocess_run):
        model_file = tmp_path / "test-model.gguf"
        model_file.touch()

        from tests.conftest import SAMPLE_PRESETS

        patched = {**SAMPLE_PRESETS, "test-local": {**SAMPLE_PRESETS["test-local"], "path": str(model_file)}}
        presets_file.write_text(json.dumps(patched))

        result = runner.invoke(app, ["switch", "test-local"])
        assert result.exit_code == 0
        assert "test-local" in result.output

    @pytest.mark.unit
    def test_switch_local_missing(self, presets_file, models_ini):
        result = runner.invoke(app, ["switch", "test-local"])
        assert result.exit_code == 1
        assert "not found" in result.output


# -- _immortalctl ---------------------------------------------------------


class TestImmortalctl:
    @pytest.mark.unit
    def test_prepends_sudo_when_needed(self):
        with (
            patch.object(llamactl, "_needs_sudo", return_value=True),
            patch.object(llamactl.subprocess, "run") as mock_run,
        ):
            mock_run.return_value = MagicMock(stdout="ok", returncode=0)
            llamactl._immortalctl("status", "llama-server")
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "sudo"
            assert str(llamactl.IMMORTALCTL_BIN) in cmd

    @pytest.mark.unit
    def test_no_sudo_when_not_needed(self):
        with (
            patch.object(llamactl, "_needs_sudo", return_value=False),
            patch.object(llamactl.subprocess, "run") as mock_run,
        ):
            mock_run.return_value = MagicMock(stdout="ok", returncode=0)
            llamactl._immortalctl("status", "llama-server")
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == str(llamactl.IMMORTALCTL_BIN)


# -- NIAH helpers ----------------------------------------------------------


class TestBuildNiahPrompt:
    @pytest.mark.unit
    def test_needle_present_in_output(self):
        prompt = llamactl._build_niah_prompt(depth_pct=50, target_chars=2000)
        assert llamactl.NIAH_NEEDLE in prompt

    @pytest.mark.unit
    def test_depth_zero_puts_needle_at_start(self):
        prompt = llamactl._build_niah_prompt(depth_pct=0, target_chars=2000)
        idx = prompt.index(llamactl.NIAH_NEEDLE)
        # Needle should appear in the first 10% of the text
        assert idx < len(prompt) * 0.1

    @pytest.mark.unit
    def test_depth_100_puts_needle_at_end(self):
        prompt = llamactl._build_niah_prompt(depth_pct=100, target_chars=2000)
        idx = prompt.index(llamactl.NIAH_NEEDLE)
        assert idx > len(prompt) * 0.7

    @pytest.mark.unit
    def test_custom_needle(self):
        custom = "The secret code is 7249."
        prompt = llamactl._build_niah_prompt(depth_pct=50, target_chars=2000, needle=custom)
        assert custom in prompt
        assert llamactl.NIAH_NEEDLE not in prompt


class TestScoreNiahResponse:
    @pytest.mark.unit
    def test_pass_when_keywords_present(self):
        response = "You should eat a sandwich and sit in Dolores Park on a sunny day."
        assert llamactl._score_niah_response(response) is True

    @pytest.mark.unit
    def test_fail_when_keywords_missing(self):
        response = "I'm not sure what the best thing to do is."
        assert llamactl._score_niah_response(response) is False

    @pytest.mark.unit
    def test_case_insensitive(self):
        response = "Eat a SANDWICH at DOLORES PARK on a SUNNY day."
        assert llamactl._score_niah_response(response) is True

    @pytest.mark.unit
    def test_custom_keywords(self):
        response = "The code is 7249."
        assert llamactl._score_niah_response(response, keywords=["7249"]) is True


class TestNiahCommand:
    @pytest.mark.unit
    def test_niah_pass(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Eat a sandwich and sit in Dolores Park on a sunny day."}}]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch.object(llamactl.httpx, "post", return_value=mock_resp) as mock_post:
            result = runner.invoke(app, ["niah"])
        assert result.exit_code == 0
        assert "PASS" in result.output
        mock_post.assert_called_once()

    @pytest.mark.unit
    def test_niah_fail(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "I don't know."}}]}
        mock_resp.raise_for_status = MagicMock()
        with patch.object(llamactl.httpx, "post", return_value=mock_resp):
            result = runner.invoke(app, ["niah"])
        assert result.exit_code == 0
        assert "FAIL" in result.output

    @pytest.mark.unit
    def test_niah_server_down(self):
        with patch.object(llamactl.httpx, "post", side_effect=httpx.ConnectError("refused")):
            result = runner.invoke(app, ["niah"])
        assert result.exit_code == 1

    @pytest.mark.unit
    def test_niah_http_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = '{"error": "context too small"}'
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Bad Request", request=MagicMock(), response=mock_resp
        )
        with patch.object(llamactl.httpx, "post", return_value=mock_resp):
            result = runner.invoke(app, ["niah"])
        assert result.exit_code == 1

    @pytest.mark.unit
    def test_niah_custom_depth_and_context(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "sandwich dolores park sunny"}}]}
        mock_resp.raise_for_status = MagicMock()
        with patch.object(llamactl.httpx, "post", return_value=mock_resp):
            result = runner.invoke(app, ["niah", "--depth", "25", "--context", "2048"])
        assert result.exit_code == 0
        assert "PASS" in result.output
