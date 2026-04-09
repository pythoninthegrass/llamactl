# Model setup

## HuggingFace models

Use `-hf` to pull a clean GGUF from HuggingFace on first run:

```bash
llama-server -hf ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M
```

This downloads and caches the model under
`~/.cache/huggingface/hub/models--<org>--<repo>/`. Subsequent runs
reuse the cache.

Create a symlink in `~/models/` pointing to the cached file:

```bash
ln -s ~/.cache/huggingface/hub/models--ggml-org--gemma-4-E4B-it-GGUF/snapshots/<hash>/gemma-4-e4b-it-Q4_K_M.gguf \
  ~/models/gemma4-e4b.gguf
```

The symlink is referenced in `presets.json` via the `model_file` key
and in `models.ini` via `model = /Users/lance/models/<name>.gguf`.

## Ollama models

Pull via `lctl` or the Ollama CLI:

```bash
lctl pull gpt-oss:20b
# or
ollama pull gpt-oss:20b
```

`lctl switch` resolves Ollama models by reading the manifest at
`~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>`
and following the digest to the blob file.

## Ollama blob caveat

Some Ollama blobs have malformed GGUF headers (e.g., `gemma4:e4b`
declares 2131 tensors but the file contains 720). llama-server
cannot load these. Use the HuggingFace source instead.

## models.ini

`lctl switch` generates `~/models/models.ini` from the selected preset:

```ini
version = 1

[*]
c = 32768
n-gpu-layers = 99

[gemma4]
model = /Users/lance/models/gemma4-e4b.gguf
jinja = true
c = 32768
```

The `[*]` section sets defaults. Each named section defines a model
preset that llama-server registers in router mode.

## Memory budget

On 16 GB unified memory with the 4.95 GB Q4_K_M Gemma 4 E4B model,
32768 context tokens is a practical upper bound. The model's native
context is 131072 but that exceeds available memory.

Router mode can load multiple models simultaneously. On 16 GB, avoid
requesting more than one model concurrently or the GPU will OOM.
