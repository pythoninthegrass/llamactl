"""Shared fixtures for llamactl tests."""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path so we can import llamactl
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llamactl

SAMPLE_PRESETS = {
    "test-hf": {
        "source": "hf",
        "hf_repo": "test-org/test-model-GGUF:Q4_K_M",
        "model_file": "test-model.gguf",
        "jinja": True,
        "context": 16384,
    },
    "test-ollama": {
        "source": "ollama",
        "ollama_model": "test-model:7b",
        "context": 32768,
    },
    "test-local": {
        "source": "local",
        "path": "/tmp/test-model.gguf",
        "context": 8192,
    },
}

SAMPLE_MODELS_RESPONSE = {
    "data": [
        {"id": "model-a", "object": "model"},
        {"id": "model-b", "object": "model"},
    ]
}

SAMPLE_OLLAMA_MANIFEST = {
    "layers": [
        {
            "mediaType": "application/vnd.ollama.image.model",
            "digest": "sha256:abc123def456",
        },
        {
            "mediaType": "application/vnd.ollama.image.license",
            "digest": "sha256:lic789",
        },
    ]
}


@pytest.fixture()
def presets_file(tmp_path):
    """Write sample presets to a temp file and patch PRESETS_PATH."""
    p = tmp_path / "presets.json"
    p.write_text(json.dumps(SAMPLE_PRESETS))
    with patch.object(llamactl, "PRESETS_PATH", p):
        yield p


@pytest.fixture()
def models_ini(tmp_path):
    """Provide a temp models.ini path and patch MODELS_INI_PATH."""
    p = tmp_path / "models.ini"
    with patch.object(llamactl, "MODELS_INI_PATH", p):
        yield p


@pytest.fixture()
def ollama_tree(tmp_path):
    """Build a minimal Ollama directory tree with a manifest and blob."""
    models_dir = tmp_path / "ollama_models"
    manifest_dir = models_dir / "manifests" / "registry.ollama.ai" / "library" / "test-model"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "7b").write_text(json.dumps(SAMPLE_OLLAMA_MANIFEST))

    blobs_dir = models_dir / "blobs"
    blobs_dir.mkdir()
    blob = blobs_dir / "sha256-abc123def456"
    blob.write_bytes(b"\x00" * 64)

    with patch.object(llamactl, "OLLAMA_MODELS_DIR", models_dir):
        yield models_dir


@pytest.fixture()
def mock_immortalctl():
    """Patch _immortalctl to avoid real subprocess calls."""
    with patch.object(llamactl, "_immortalctl") as m:
        m.return_value = MagicMock(stdout="", stderr="", returncode=0)
        yield m


@pytest.fixture()
def mock_subprocess_run():
    """Patch subprocess.run globally within llamactl."""
    with patch.object(llamactl.subprocess, "run") as m:
        m.return_value = MagicMock(stdout="", stderr="", returncode=0)
        yield m


@pytest.fixture()
def mock_httpx_get():
    """Patch httpx.get within llamactl."""
    with patch.object(llamactl.httpx, "get") as m:
        resp = MagicMock()
        resp.json.return_value = SAMPLE_MODELS_RESPONSE
        m.return_value = resp
        yield m
