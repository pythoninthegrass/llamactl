# llamactl

CLI for managing a local [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
server on macOS Apple Silicon. Wraps [immortal](https://immortal.run/)
process supervision and handles model switching across HuggingFace,
Ollama, and local GGUF sources.

## Requirements

Minimum:

- macOS with Apple Silicon (M-series)
- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) (for PEP 723 script execution)
- [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) built with Metal
- [immortal](https://immortal.run/) configured as a LaunchDaemon

Recommended:

- 16 GB+ unified memory (32 GB for larger quants or multiple models)
- [mise](https://mise.jdx.dev/) for managing uv and Python runtimes
- [Ollama](https://ollama.com/) for pulling models
- [OpenCode](https://opencode.ai/) or any OpenAI-compatible client

## Install

```bash
git clone <this-repo> ~/git/llamactl
chmod +x ~/git/llamactl/main.py
ln -s ~/git/llamactl/main.py ~/.local/bin/lctl
```

Verify:

```bash
lctl --help
```

On first run, uv creates a cached virtual environment and installs
dependencies (typer, jinja2, ollama, httpx). Subsequent runs are instant.

## Quickstart

```bash
# Check server status
lctl status

# List loaded models
lctl models

# List available presets
lctl presets

# Switch to a preset (regenerates models.ini, restarts server)
lctl switch gemma4

# Pull a model from Ollama
lctl pull gpt-oss:20b

# Pull from HuggingFace
lctl pull --hf ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M

# Follow server logs
lctl logs -f

# Show stderr log
lctl logs --err

# Start / stop / restart
lctl start
lctl stop
lctl restart
```

The `switch` command prints the corresponding `opencode.jsonc` provider
config to stdout after restarting.

## Presets

Model presets are defined in `presets.json` alongside the script:

```json
{
  "gemma4": {
    "source": "hf",
    "hf_repo": "ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M",
    "model_file": "gemma4-e4b.gguf",
    "jinja": true,
    "context": 32768
  },
  "gpt-oss:20b": {
    "source": "ollama",
    "ollama_model": "gpt-oss:20b",
    "context": 32768
  }
}
```

Sources:

| Source | Resolution |
|--------|-----------|
| `hf` | Symlink in `~/models/` pointing to HF cache. Set via `model_file`. |
| `ollama` | Resolved from Ollama manifest at `~/.ollama/models/manifests/`. |
| `local` | Direct path to a `.gguf` file via `path` key. |

## Setup guides

- [Building llama-cpp-turboquant](docs/build.md)
- [Model setup](docs/models.md)
- [Immortal process supervision](docs/immortal.md)
- [OpenCode configuration](docs/opencode.md)
- [Troubleshooting](docs/troubleshooting.md)
