# Troubleshooting

## `model X not found` (400)

The model ID in the opencode config does not match what llama-server
registered. Check with:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq .data[].id
```

## `wrong number of tensors`

The GGUF file has a corrupt header. Use the HuggingFace source instead:

```bash
lctl pull --hf ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M
```

## Multiple opencode configs

System-level `/Library/Application Support/opencode/opencode.jsonc`
overrides `~/.config/opencode/opencode.jsonc`. Check both.

## Context too small

Increase `context` in `presets.json`. Max for Gemma 4 E4B is 131072.
On 16 GB RAM with the 4.95 GB Q4_K_M model, 32768 is a practical
upper bound.

## GPU out of memory

Router mode can load multiple model instances simultaneously. On 16 GB,
a second model request while one is already loaded will OOM the GPU.

Fix: `lctl restart` kills all processes (router + children) and lets
immortal start fresh with only the router.

## Services crash-loop silently

If `immortaldir.log` shows repeated "Starting: llama-server" with no
llama-server log output:

1. Check the `EnvironmentVariables/PATH` in the LaunchDaemon plist
   includes `/opt/homebrew/bin`. Without it, `immortaldir` cannot find
   the `immortal` binary.
2. Check `/usr/local/var/log` is writable by the service user.
3. Test manually: `sudo immortal -c /usr/local/etc/immortal/llama-server.yml`

## Metal GPU falls back to CPU

A LaunchDaemon runs outside the GUI session. Add `SessionCreate` to the
plist. See [immortal docs](immortal.md#if-metal-gpu-fails-to-initialize).

## `immortalctl` shows empty output

The control socket at `/var/run/immortal/llama-server/immortal.sock` is
root-owned. Use `sudo immortalctl` or `lctl status` (which adds sudo
automatically).
