"""Integration tests for llamactl -- multi-component flows with mocked system boundaries."""

import httpx
import json
import main
import pytest
from main import app
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch


def _parse_models_ini(path):
    """Parse models.ini which has a bare 'version = 1' before any section header."""
    import configparser

    text = path.read_text()
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


# -- switch full pipeline ---------------------------------------------------


class TestSwitchPipelineHF:
    """switch with an HF preset: load preset -> resolve symlink -> write ini -> restart -> opencode output."""

    @pytest.mark.integration
    def test_writes_valid_ini_and_restarts(self, presets_file, models_ini, mock_subprocess_run):
        model_file = models_ini.parent / "test-model.gguf"
        model_file.touch()

        result = runner.invoke(app, ["switch", "test-hf"])

        assert result.exit_code == 0
        # models.ini written and parseable
        ini = _parse_models_ini(models_ini)
        assert "test-hf" in ini.sections()
        assert ini["test-hf"]["c"] == "16384"
        assert ini["test-hf"]["model"] == str(model_file)
        assert ini["test-hf"]["jinja"] == "true"
        # pkill issued for restart
        mock_subprocess_run.assert_called_once_with(["pkill", "-f", "llama-server"], capture_output=True)

    @pytest.mark.integration
    def test_opencode_json_in_output(self, presets_file, models_ini, mock_subprocess_run):
        model_file = models_ini.parent / "test-model.gguf"
        model_file.touch()

        result = runner.invoke(app, ["switch", "test-hf"])

        assert result.exit_code == 0
        # Output contains valid opencode JSON block
        lines = result.output.split("--- opencode.jsonc provider config ---")
        assert len(lines) == 2
        opencode_json = json.loads(lines[1].strip())
        provider = opencode_json["provider"]["llama.cpp"]
        assert provider["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
        assert "test-hf" in provider["models"]
        assert provider["models"]["test-hf"]["limit"]["context"] == 16384


class TestSwitchPipelineOllama:
    """switch with an Ollama preset: load preset -> resolve blob -> write ini -> restart."""

    @pytest.mark.integration
    def test_writes_ini_with_blob_path(self, presets_file, models_ini, ollama_tree, mock_subprocess_run):
        from tests.conftest import SAMPLE_PRESETS

        patched = {**SAMPLE_PRESETS, "test-ollama": {**SAMPLE_PRESETS["test-ollama"], "ollama_model": "test-model:7b"}}
        presets_file.write_text(json.dumps(patched))

        result = runner.invoke(app, ["switch", "test-ollama"])

        assert result.exit_code == 0
        ini = _parse_models_ini(models_ini)
        assert "test-ollama" in ini.sections()
        # Model path should point to the blob
        assert "sha256-abc123def456" in ini["test-ollama"]["model"]


class TestSwitchPipelineLocal:
    """switch with a local preset: load preset -> verify path -> write ini -> restart."""

    @pytest.mark.integration
    def test_writes_ini_with_local_path(self, presets_file, models_ini, tmp_path, mock_subprocess_run):
        from tests.conftest import SAMPLE_PRESETS

        model_file = tmp_path / "test-model.gguf"
        model_file.touch()
        patched = {**SAMPLE_PRESETS, "test-local": {**SAMPLE_PRESETS["test-local"], "path": str(model_file)}}
        presets_file.write_text(json.dumps(patched))

        result = runner.invoke(app, ["switch", "test-local"])

        assert result.exit_code == 0
        ini = _parse_models_ini(models_ini)
        assert "test-local" in ini.sections()
        assert ini["test-local"]["model"] == str(model_file)
        assert ini["test-local"]["c"] == "8192"
        # jinja should NOT be present (not set in preset)
        assert "jinja" not in ini["test-local"]


# -- status integration -----------------------------------------------------


class TestStatusIntegration:
    """status command: immortalctl output + httpx model listing combined."""

    @pytest.mark.integration
    def test_running_with_multiple_models(self, mock_immortalctl, mock_httpx_get):
        mock_immortalctl.return_value.stdout = "llama-server: up (pid 42) 3h"
        mock_httpx_get.return_value.json.return_value = {
            "data": [
                {"id": "gemma4", "object": "model"},
                {"id": "qwen3", "object": "model"},
            ]
        }

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "up" in result.output
        assert "gemma4" in result.output
        assert "qwen3" in result.output

    @pytest.mark.integration
    def test_running_but_no_models_loaded(self, mock_immortalctl, mock_httpx_get):
        mock_immortalctl.return_value.stdout = "llama-server: up (pid 42)"
        mock_httpx_get.return_value.json.return_value = {"data": []}

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "up" in result.output
        # No "Loaded models" line when data is empty
        assert "Loaded models" not in result.output

    @pytest.mark.integration
    def test_running_but_server_unreachable(self, mock_immortalctl):
        mock_immortalctl.return_value.stdout = "llama-server: up (pid 42)"
        with patch.object(main.httpx, "get", side_effect=httpx.ConnectError("refused")):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "up" in result.output
        assert "not responding" in result.output


# -- niah integration -------------------------------------------------------


class TestNiahIntegration:
    """niah command: models.ini parsing -> prompt build -> HTTP post -> scoring."""

    @pytest.fixture(autouse=True)
    def _niah_env(self, tmp_path):
        ini = tmp_path / "models.ini"
        ini.write_text("version = 1\n\n[*]\nc = 32768\n\n[gemma4]\nmodel = /tmp/gemma4.gguf\nc = 32768\n")
        with patch.object(main, "MODELS_INI_PATH", ini):
            yield

    @pytest.mark.integration
    def test_model_name_read_from_ini(self):
        """Verify the model name sent in the POST body matches models.ini."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "sandwich dolores park sunny"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(main.httpx, "post", return_value=mock_resp) as mock_post:
            runner.invoke(app, ["niah"])

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "gemma4"

    @pytest.mark.integration
    def test_context_flag_scales_prompt_size(self):
        """--context flag controls haystack size sent to the server."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "sandwich dolores park sunny"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        payloads = []

        def capture_post(url, json, timeout):
            payloads.append(json)
            return mock_resp

        with patch.object(main.httpx, "post", side_effect=capture_post):
            runner.invoke(app, ["niah", "--context", "1024"])
            runner.invoke(app, ["niah", "--context", "8192"])

        small_content = payloads[0]["messages"][1]["content"]
        large_content = payloads[1]["messages"][1]["content"]
        assert len(large_content) > len(small_content)

    @pytest.mark.integration
    def test_custom_needle_and_question(self):
        """Custom needle/question are threaded through to the request and scoring."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "The launch code is 7249."}}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(main.httpx, "post", return_value=mock_resp) as mock_post:
            result = runner.invoke(app, [
                "niah",
                "--needle", "The launch code is 7249.",
                "--question", "What is the launch code?",
            ])

        assert result.exit_code == 0
        assert "PASS" in result.output
        payload = mock_post.call_args[1]["json"]
        user_msg = payload["messages"][1]["content"]
        assert "The launch code is 7249." in user_msg
        assert "What is the launch code?" in user_msg

    @pytest.mark.integration
    def test_niah_no_model_section_in_ini(self, tmp_path):
        """Exit 1 when models.ini has no model section (only [*])."""
        ini = tmp_path / "models.ini"
        ini.write_text("version = 1\n\n[*]\nc = 32768\n")
        with patch.object(main, "MODELS_INI_PATH", ini):
            result = runner.invoke(app, ["niah"])

        assert result.exit_code == 1
        assert "Could not determine model name" in result.output

    @pytest.mark.integration
    def test_niah_missing_ini(self, tmp_path):
        """Exit 1 when models.ini does not exist."""
        with patch.object(main, "MODELS_INI_PATH", tmp_path / "nonexistent.ini"):
            result = runner.invoke(app, ["niah"])

        assert result.exit_code == 1


# -- pull integration -------------------------------------------------------


class TestPullOllamaIntegration:
    """pull --ollama: streams progress from ollama client, resolves blob path."""

    @pytest.mark.integration
    def test_pull_streams_progress_and_resolves_blob(self, ollama_tree):
        progress_events = [
            MagicMock(status="downloading", total=1000, completed=500),
            MagicMock(status="downloading", total=1000, completed=1000),
            MagicMock(status="success", total=None, completed=None),
        ]

        with patch("main.ollama_client", create=True) as _:
            with patch.dict("sys.modules", {"ollama": MagicMock()}) as _:
                import sys

                mock_ollama = sys.modules["ollama"]
                mock_ollama.pull.return_value = iter(progress_events)

                # Patch the import inside pull()
                with patch("main._resolve_ollama_model_path") as mock_resolve:
                    blob = ollama_tree / "blobs" / "sha256-abc123def456"
                    mock_resolve.return_value = blob

                    result = runner.invoke(app, ["pull", "test-model:7b"])

        assert result.exit_code == 0


class TestPullHFIntegration:
    """pull --hf: launches llama-server subprocess, reads stdout until model loaded."""

    @pytest.mark.integration
    def test_pull_terminates_after_model_loaded(self):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([
            "loading model...\n",
            "model loaded successfully\n",
        ])
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()

        with patch.object(main.subprocess, "Popen", return_value=mock_proc):
            result = runner.invoke(app, ["pull", "--hf", "org/model-GGUF:Q4_K_M"])

        assert result.exit_code == 0
        mock_proc.terminate.assert_called_once()

    @pytest.mark.integration
    def test_pull_terminates_on_listening(self):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([
            "initializing...\n",
            "listening on 0.0.0.0:18080\n",
        ])
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()

        with patch.object(main.subprocess, "Popen", return_value=mock_proc):
            result = runner.invoke(app, ["pull", "--hf", "org/model-GGUF:Q4_K_M"])

        assert result.exit_code == 0
        mock_proc.terminate.assert_called_once()


# -- logs integration -------------------------------------------------------


class TestLogsIntegration:
    """logs command: file checks and flag handling."""

    @pytest.mark.integration
    def test_logs_missing_file(self, tmp_path):
        with patch.object(main, "LOG_PATH", tmp_path / "nope.log"):
            result = runner.invoke(app, ["logs"])

        assert "not found" in result.output

    @pytest.mark.integration
    def test_logs_err_flag_missing_file(self, tmp_path):
        with patch.object(main, "STDERR_LOG_PATH", tmp_path / "nope-err.log"):
            result = runner.invoke(app, ["logs", "--err"])

        assert "not found" in result.output

    @pytest.mark.integration
    def test_logs_selects_stderr_file(self, tmp_path):
        stdout_log = tmp_path / "stdout.log"
        stderr_log = tmp_path / "stderr.log"
        stdout_log.write_text("stdout content")
        stderr_log.write_text("stderr content")

        with (
            patch.object(main, "LOG_PATH", stdout_log),
            patch.object(main, "STDERR_LOG_PATH", stderr_log),
            patch.object(main.os, "execvp") as mock_exec,
        ):
            runner.invoke(app, ["logs", "--err"])

        # execvp called with the stderr log path
        cmd = mock_exec.call_args[0][1]
        assert str(stderr_log) in cmd

    @pytest.mark.integration
    def test_logs_follow_flag(self, tmp_path):
        log = tmp_path / "server.log"
        log.write_text("some log line")

        with (
            patch.object(main, "LOG_PATH", log),
            patch.object(main.os, "execvp") as mock_exec,
        ):
            runner.invoke(app, ["logs", "-f"])

        cmd = mock_exec.call_args[0][1]
        assert "-f" in cmd


# -- models integration -----------------------------------------------------


class TestModelsIntegration:
    """models command: httpx response parsing."""

    @pytest.mark.integration
    def test_models_empty_list(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        with patch.object(main.httpx, "get", return_value=mock_resp):
            result = runner.invoke(app, ["models"])

        assert result.exit_code == 0
        # No model IDs printed, but no error either
        assert result.output.strip() == ""

    @pytest.mark.integration
    def test_models_multiple(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"id": "gemma4", "object": "model"},
                {"id": "qwen3-coder:30b", "object": "model"},
            ]
        }
        with patch.object(main.httpx, "get", return_value=mock_resp):
            result = runner.invoke(app, ["models"])

        assert result.exit_code == 0
        lines = [l for l in result.output.strip().splitlines() if l.strip()]
        assert lines == ["gemma4", "qwen3-coder:30b"]
