# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

llamactl is a single-file Python CLI (`src/llamactl/main.py`) for managing a local
[llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) server
on macOS Apple Silicon. It wraps [immortal](https://immortal.run/) process
supervision and handles model switching across HuggingFace, Ollama, and local
GGUF sources.

The script uses PEP 723 inline metadata so `uv run --script` resolves
dependencies automatically -- there is no separate `requirements.txt`.

## Commands

```bash
# Run the CLI (symlinked as lctl)
uv run --script main.py --help

# Lint / format
ruff check .
ruff format --check --diff .
ruff format .

# Run tests
uv run pytest                     # all tests
uv run pytest -k test_name        # single test
uv run pytest -m unit             # by marker

# Install dev deps into a venv
uv sync --all-extras
```

## Architecture

- **`src/llamactl/main.py`** -- entire CLI in one file (`main.py` at root is a symlink). Uses Typer for subcommands
  (`status`, `start`, `stop`, `restart`, `logs`, `models`, `switch`, `pull`,
  `presets`). Hardcoded paths assume macOS Homebrew layout and `~/models/`,
  `~/git/llama-cpp-turboquant/`, `~/.ollama/models/`.
- **`presets.json`** -- model preset definitions (source type, repo/path,
  context length, jinja flag). Three source types: `hf`, `ollama`, `local`.
- **`models.ini`** -- generated at `~/models/models.ini` by the `switch`
  command using a Jinja2 template. Consumed by llama-server.
- **Immortal** -- llama-server runs as a LaunchDaemon under immortal. The CLI
  shells out to `immortalctl` (with sudo when the socket is root-owned) for
  start/stop/status, and uses `pkill -f llama-server` for hard restarts.
- **`docs/`** -- setup guides for building llama-cpp, model management,
  immortal config, OpenCode integration, and troubleshooting.

## Key conventions

- Python 3.13, ruff for linting/formatting (line-length 130, spaces, LF)
- pytest markers: `unit`, `integration`, `e2e`, `benchmark`
- Dependencies: httpx, jinja2, ollama, typer

## Context7

Always use Context7 MCP when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.

### Libraries

- anomalyco/opencode
- encode/httpx
- fastapi/typer
- huggingface/huggingface_hub
- immortal/immortal
- jdx/mise
- mrlesk/backlog.md
- ollama/ollama-python
- thetom/turboquant_plus
- websites/taskfile_dev
