#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "httpx>=0.27,<1.0",
#   "jinja2>=3.1.6",
#   "ollama>=0.4",
#   "typer>=0.24.1",
# ]
# [tool.uv]
# exclude-newer = "2026-04-30T00:00:00Z"
# ///

# pyright: reportMissingImports=false

"""
Usage:
    llamactl <status|start|stop|restart|logs|models|switch|pull|presets|niah>

Commands:
    status:     Show immortal service status and loaded models
    start:      Start llama-server
    stop:       Stop llama-server
    restart:    Restart llama-server (kills all child model processes too)
    logs:       Tail llama-server logs
    models:     List models available on the server
    switch:     Switch to a model preset and restart the server
    pull:       Download a model from HuggingFace or Ollama
    presets:    List available model presets
    niah:       Run a Needle-In-A-Haystack retrieval test

Note:
    Manages llama-server via immortal.
"""

import contextlib
import httpx
import jinja2
import json
import os
import subprocess
import time
import typer
from pathlib import Path
from typing import Annotated

SCRIPT_DIR = Path(__file__).resolve().parent
PRESETS_PATH = SCRIPT_DIR / "presets.json"
MODELS_INI_PATH = Path.home() / "models" / "models.ini"
LLAMA_SERVER_BIN = Path.home() / "git" / "llama-cpp-turboquant" / "build" / "bin" / "llama-server"
OLLAMA_MODELS_DIR = Path.home() / ".ollama" / "models"
IMMORTALCTL_BIN = Path("/opt/homebrew/bin/immortalctl")
IMMORTAL_SOCKET = Path("/var/run/immortal/llama-server/immortal.sock")
LOG_PATH = Path("/usr/local/var/log/llama-server.log")
STDERR_LOG_PATH = Path("/usr/local/var/log/llama-server-err.log")
SERVER_URL = "http://127.0.0.1:8080"
SERVICE_NAME = "llama-server"

OPENCODE_TEMPLATE = jinja2.Template("""\
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama.cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (local)",
      "options": {
        "baseURL": "{{ server_url }}/v1"
      },
      "models": {
        "{{ model_id }}": {
          "name": "{{ display_name }}",
          "limit": {
            "context": {{ context }},
            "output": 8192
          }
        }
      }
    }
  }
}
""")

MODELS_INI_TEMPLATE = jinja2.Template("""\
version = 1

[*]
c = {{ context }}
n-gpu-layers = 99

[{{ name }}]
model = {{ model_path }}
{% if jinja %}jinja = true
{% endif %}c = {{ context }}
""")

app = typer.Typer(help=__doc__, invoke_without_command=True)


@app.callback()
def main(ctx: typer.Context):
    """Manage llama-server via immortal."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


def _needs_sudo() -> bool:
    """Check if immortalctl needs sudo (socket is root-owned)."""
    if IMMORTAL_SOCKET.exists():
        return not os.access(IMMORTAL_SOCKET, os.R_OK)
    return True


def _immortalctl(*args: str) -> subprocess.CompletedProcess:
    """Run immortalctl with sudo if needed."""
    cmd = [str(IMMORTALCTL_BIN), *args]
    if _needs_sudo():
        cmd = ["sudo", *cmd]
    return subprocess.run(cmd, capture_output=True, text=True)


def _load_presets() -> dict:
    """Load model presets from presets.json."""
    if not PRESETS_PATH.exists():
        typer.echo(f"Presets file not found: {PRESETS_PATH}", err=True)
        raise typer.Exit(1)
    return json.loads(PRESETS_PATH.read_text())


def _resolve_ollama_model_path(model_name: str) -> Path | None:
    """Resolve an Ollama model name to its GGUF blob path."""
    parts = model_name.split(":")
    name = parts[0]
    tag = parts[1] if len(parts) > 1 else "latest"

    manifest_path = OLLAMA_MODELS_DIR / "manifests" / "registry.ollama.ai" / "library" / name / tag
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text())
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer["digest"]
            blob_name = digest.replace(":", "-")
            return OLLAMA_MODELS_DIR / "blobs" / blob_name

    return None


@app.command()
def status():
    """Show immortal service status and loaded models."""
    result = _immortalctl("status", SERVICE_NAME)
    if result.stdout.strip():
        typer.echo(result.stdout.strip())
    elif result.returncode != 0:
        typer.echo("llama-server is not running")
        return

    try:
        resp = httpx.get(f"{SERVER_URL}/v1/models", timeout=5)
        model_ids = [m["id"] for m in resp.json().get("data", [])]
        if model_ids:
            typer.echo(f"\nLoaded models: {', '.join(model_ids)}")
    except httpx.ConnectError:
        typer.echo("\nServer not responding on port 8080")


@app.command()
def start():
    """Start llama-server."""
    result = _immortalctl("start", SERVICE_NAME)
    typer.echo(result.stdout.strip() or result.stderr.strip() or "Started")


@app.command()
def stop():
    """Stop llama-server."""
    result = _immortalctl("stop", SERVICE_NAME)
    typer.echo(result.stdout.strip() or result.stderr.strip() or "Stopped")


@app.command()
def restart():
    """Restart llama-server (kills all child model processes too)."""
    # In router mode, immortalctl restart only cycles the router process.
    # Child model servers survive. Kill the whole process tree first.
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(2)
    typer.echo("Restarted (immortal auto-recovers)")


@app.command()
def logs(
    follow: Annotated[bool, typer.Option("-f", "--follow", help="Follow log output")] = False,
    err: Annotated[bool, typer.Option("--err", help="Show stderr log instead")] = False,
):
    """Show llama-server logs."""
    log_file = STDERR_LOG_PATH if err else LOG_PATH
    if not log_file.exists():
        typer.echo(f"Log file not found: {log_file}", err=True)
        return
    cmd = ["tail"]
    if follow:
        cmd.append("-f")
    else:
        cmd.extend(["-n", "50"])
    cmd.append(str(log_file))
    os.execvp("tail", cmd)


@app.command()
def models():
    """List models available on the server."""
    with contextlib.suppress(httpx.ConnectError):
        resp = httpx.get(f"{SERVER_URL}/v1/models", timeout=5)
        data = resp.json().get("data", [])
        for model in data:
            typer.echo(model["id"])
        return
    typer.echo("Server not responding on port 8080", err=True)
    raise typer.Exit(1)


@app.command()
def switch(preset: str):
    """Switch to a model preset and restart the server."""
    all_presets = _load_presets()
    if preset not in all_presets:
        typer.echo(f"Unknown preset: {preset}", err=True)
        typer.echo(f"Available: {', '.join(all_presets)}", err=True)
        raise typer.Exit(1)

    cfg = all_presets[preset]
    source = cfg["source"]
    model_path = None

    if source == "hf":
        # HF models are accessed via a symlink in ~/models/ that points
        # to the HF cache. This avoids router mode loading duplicates.
        model_file = cfg.get("model_file", f"{preset}.gguf")
        symlink = MODELS_INI_PATH.parent / model_file
        if symlink.exists():
            model_path = symlink
        else:
            typer.echo(f"Model file not found: {symlink}", err=True)
            typer.echo(f"Create it: ln -s <cached-gguf-path> {symlink}", err=True)
            raise typer.Exit(1)
    elif source == "ollama":
        model_path = _resolve_ollama_model_path(cfg["ollama_model"])
        if model_path is None or not model_path.exists():
            typer.echo(f"Ollama model not found locally: {cfg['ollama_model']}", err=True)
            typer.echo(f"Run: lctl pull --ollama {cfg['ollama_model']}", err=True)
            raise typer.Exit(1)
    elif source == "local":
        model_path = Path(cfg["path"])
        if not model_path.exists():
            typer.echo(f"Model file not found: {model_path}", err=True)
            raise typer.Exit(1)

    ini_content = MODELS_INI_TEMPLATE.render(
        name=preset,
        source=source,
        context=cfg.get("context", 32768),
        jinja=cfg.get("jinja", False),
        hf_repo=cfg.get("hf_repo", ""),
        model_path=str(model_path) if model_path else "",
    )

    MODELS_INI_PATH.write_text(ini_content)
    typer.echo(f"Wrote {MODELS_INI_PATH}")
    typer.echo(f"Preset: {preset} ({source})")

    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(2)
    typer.echo("Restarted (immortal auto-recovers)")

    model_id = preset
    typer.echo("\n--- opencode.jsonc provider config ---")
    typer.echo(
        OPENCODE_TEMPLATE.render(
            server_url=SERVER_URL,
            model_id=model_id,
            display_name=preset,
            context=cfg.get("context", 32768),
        )
    )


@app.command()
def pull(
    model: str,
    hf: Annotated[bool, typer.Option("--hf", help="Pull from HuggingFace (via llama-server -hf)")] = False,
    ollama: Annotated[bool, typer.Option("--ollama", help="Pull from Ollama (default)")] = True,
):
    """Pull a model from HuggingFace or Ollama."""
    source = "hf" if hf else "ollama"

    if source == "ollama":
        import ollama as ollama_client

        typer.echo(f"Pulling {model} from Ollama...")
        for progress in ollama_client.pull(model, stream=True):
            if progress.total:
                pct = (progress.completed or 0) / progress.total * 100
                print(f"\r  {progress.status}: {pct:.0f}%", end="", flush=True)
            else:
                print(f"\r  {progress.status}", end="", flush=True)
        print()

        blob_path = _resolve_ollama_model_path(model)
        if blob_path and blob_path.exists():
            typer.echo(f"Model stored at: {blob_path}")
        else:
            typer.echo("Pull completed but could not resolve blob path", err=True)

    elif source == "hf":
        typer.echo(f"Pulling {model} from HuggingFace via llama-server...")
        typer.echo("This starts llama-server temporarily to download the model.")
        proc = subprocess.Popen(
            [str(LLAMA_SERVER_BIN), "-hf", model, "-ngl", "0", "-c", "512", "--port", "18080"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in proc.stdout:
                typer.echo(f"  {line}", nl=False)
                if "model loaded" in line.lower() or "listening" in line.lower():
                    typer.echo("\nDownload complete.")
                    break
        finally:
            proc.terminate()
            proc.wait()


NIAH_NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
NIAH_QUESTION = "What is the best thing to do in San Francisco?"
NIAH_KEYWORDS = ["sandwich", "dolores park", "sunny"]

NIAH_FILLER_PARAGRAPHS = [
    "The history of urban planning in modern cities reveals a complex interplay between governmental policy and private development. Zoning regulations, first introduced in the early twentieth century, sought to separate residential areas from industrial zones. Over time, mixed-use development gained popularity as city planners recognized the benefits of walkable neighborhoods. Transportation infrastructure, including highways and public transit systems, shaped the growth patterns of metropolitan areas across the country.",
    "Agricultural practices have evolved significantly over the past century. The introduction of mechanized farming equipment transformed rural economies and reduced the labor required to produce staple crops. Crop rotation techniques, developed over centuries of trial and error, help maintain soil fertility and reduce pest populations. Modern irrigation systems allow farming in regions that would otherwise be too arid for cultivation, expanding the global food supply.",
    "The development of telecommunications technology has proceeded through several distinct phases. Early telegraph systems enabled long-distance communication for the first time, followed by telephone networks that connected homes and businesses. The advent of radio broadcasting created new forms of entertainment and information distribution. Television further transformed media consumption patterns, while satellite communications enabled global connectivity.",
    "Marine biology encompasses the study of organisms living in ocean environments, from microscopic plankton to the largest whales. Coral reef ecosystems support an extraordinary diversity of species and play a crucial role in coastal protection. Deep-sea exploration has revealed previously unknown species adapted to extreme pressure and darkness. Ocean currents influence global climate patterns and distribute nutrients across vast distances.",
    "The principles of thermodynamics govern energy transfer in physical systems. The first law establishes that energy cannot be created or destroyed, only converted between forms. Heat engines convert thermal energy into mechanical work with efficiencies limited by the Carnot cycle. Entropy, a measure of disorder in a system, tends to increase over time in isolated systems according to the second law.",
    "Classical music composition follows established formal structures that have evolved over centuries. Sonata form, developed during the Classical period, provides a framework for the development of musical themes. Orchestration techniques determine how musical ideas are distributed among different instrument groups. Counterpoint, the art of combining independent melodic lines, reached its highest development in the Baroque era.",
    "Geological processes operate on timescales ranging from seconds to billions of years. Plate tectonics drives the movement of continental landmasses and the formation of mountain ranges. Volcanic activity releases gases and minerals from the Earth's interior, influencing atmospheric composition. Erosion by wind and water gradually reshapes landscapes, creating valleys, canyons, and sedimentary deposits.",
    "The study of linguistics examines the structure, history, and social context of human languages. Phonology analyzes the sound systems used in different languages, while morphology studies word formation. Syntax describes the rules governing sentence structure, and semantics deals with meaning. Historical linguistics traces the evolution of language families and the spread of linguistic innovations across populations.",
]

CHARS_PER_TOKEN = 4


def _build_niah_prompt(depth_pct: int, target_chars: int, needle: str | None = None) -> str:
    """Build a haystack of filler text with a needle inserted at the given depth."""
    needle = needle or NIAH_NEEDLE
    filler_target = target_chars - len(needle)
    if filler_target < 0:
        filler_target = 0

    # Repeat filler paragraphs until we reach the target length
    paragraphs = []
    total = 0
    i = 0
    while total < filler_target:
        p = NIAH_FILLER_PARAGRAPHS[i % len(NIAH_FILLER_PARAGRAPHS)]
        paragraphs.append(p)
        total += len(p) + 2  # account for paragraph separator
        i += 1

    # Insert needle at the specified depth
    depth_pct = max(0, min(100, depth_pct))
    insert_idx = int(len(paragraphs) * depth_pct / 100)
    paragraphs.insert(insert_idx, needle)

    return "\n\n".join(paragraphs)


def _score_niah_response(response: str, keywords: list[str] | None = None) -> bool:
    """Check if the response contains the expected keywords."""
    keywords = keywords or NIAH_KEYWORDS
    response_lower = response.lower()
    return all(kw.lower() in response_lower for kw in keywords)


@app.command()
def niah(
    depth: Annotated[int, typer.Option("--depth", "-d", help="Needle depth percentage (0=start, 100=end)")] = 50,
    context: Annotated[int, typer.Option("--context", "-c", help="Target context length in tokens")] = 4096,
    needle: Annotated[str | None, typer.Option("--needle", help="Custom needle text")] = None,
    question: Annotated[str | None, typer.Option("--question", help="Custom retrieval question")] = None,
):
    """Run a Needle-In-A-Haystack retrieval test against the running server."""
    target_chars = context * CHARS_PER_TOKEN
    haystack = _build_niah_prompt(depth_pct=depth, target_chars=target_chars, needle=needle)

    q = question or NIAH_QUESTION
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question based only on the context provided."},
        {"role": "user", "content": f"{haystack}\n\nBased on the text above, {q}"},
    ]

    typer.echo(f"NIAH test: depth={depth}%, context~{context} tokens ({len(haystack)} chars)")

    try:
        resp = httpx.post(
            f"{SERVER_URL}/v1/chat/completions",
            json={"messages": messages, "max_tokens": 256},
            timeout=120,
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        typer.echo("Server not responding on port 8080", err=True)
        raise typer.Exit(1)
    except httpx.HTTPStatusError as e:
        typer.echo(f"Server error: {e.response.status_code}", err=True)
        with contextlib.suppress(Exception):
            typer.echo(e.response.text, err=True)
        raise typer.Exit(1)

    answer = resp.json()["choices"][0]["message"]["content"]
    passed = _score_niah_response(answer, keywords=needle and [needle] or None)

    typer.echo(f"Answer: {answer.strip()}")
    typer.echo(f"Result: {'PASS' if passed else 'FAIL'}")


@app.command()
def presets():
    """List available model presets."""
    for name, cfg in _load_presets().items():
        source = cfg["source"]
        detail = cfg.get("hf_repo") or cfg.get("ollama_model") or cfg.get("path", "")
        typer.echo(f"  {name:20s} {source:8s} {detail}")


if __name__ == "__main__":
    app()
