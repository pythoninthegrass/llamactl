# OpenCode configuration

The model ID in `opencode.jsonc` must match the ID reported by
`curl http://127.0.0.1:8080/v1/models`. When using `--models-preset`,
the IDs are the section names in `models.ini`.

`lctl switch` prints the correct provider config to stdout after
switching presets.

## System-level config

`/Library/Application Support/opencode/opencode.jsonc` takes precedence
over `~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama.cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "gemma4": {
          "name": "Gemma 4 E4B (local)",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  }
}
```

## User-level config

The llama.cpp provider can coexist with Ollama or other providers in
`~/.config/opencode/opencode.jsonc`. Add the `llama.cpp` key alongside
existing providers.
