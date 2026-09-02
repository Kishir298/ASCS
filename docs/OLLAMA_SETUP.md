# Connecting qwen3-coder:30b (local Ollama) to OpenCode and the ASCS CLI

This guide takes a clean machine from nothing to a working local connection
between a **local `qwen3-coder:30b` Ollama model** and the ASCS CLI, plus the
**OpenCode** CLI, over the same Ollama server.

Recommended target: a machine with **≥24 GB RAM, ideally 32–48 GB (16GB → fallback qwen2.5-coder:14b)**. `qwen3-coder:30b`
is a ~19 GB model (Q4_K_M) that needs roughly 18–20 GB of RAM (+ 65k ctx) for inference. Low-RAM
laptops (e.g. an 8 GB Mac) will struggle — use this guide on the capable
machine (the 32 GB Windows laptop), not the dev editor.

Everything here is written for **Windows (PowerShell)**. Unix/macOS users only
differ in: creating the venv, activating it, and omitting the `$env:` prefix
(use `RISALIVE=1` in bash) — the Ollama/OpenCode steps are identical.

---

## 1. Prerequisites (Windows)

| Tool | How to install | Verify |
|---|---|---|
| Ollama | Download from <https://ollama.com/download/windows> and install. Installs as a background Windows service (auto-starts). | `ollama --version` |
| OpenCode CLI | Per <https://opencode.ai> install instructions. | `opencode --version` |
| Python 3.12+ | From python.org or `winget install Python.Python.3.12`. | `py -3 --version` |
| Git | Optional; used to clone the repo. | `git --version` |

> On Windows the local model store lives at `C:\Users\<you>\.ollama\models`.
> The Ollama service listens on `http://localhost:11434`.

---

## 2. Pull and start the model

```powershell
ollama pull qwen3-coder:30b
# fallback on 16GB: ollama pull qwen2.5-coder:14b
```

Confirm the Ollama server is reachable:

```powershell
curl.exe http://localhost:11434/api/version
```

Confirm the model is registered with Ollama. It must appear in **both** the
native API **and** the OpenAI-compatible API:

```powershell
curl.exe http://localhost:11434/v1/models     # OpenCode / @ai-sdk uses this
curl.exe http://localhost:11434/api/tags      # ASCS native client uses this
```

Both should list a `qwen3-coder:30b` entry. If `/v1/models` is empty, the
`@ai-sdk/openai-compatible` provider cannot see the model.

---

## 3. ASCS CLI setup and validation

```powershell
cd C:\path\to\ASCS
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 3.1 Check connectivity and model availability

```powershell
risa --check          # expected: server reachable, model available
risa --list-models    # expected: qwen3-coder:30b listed as INSTALLED
```

### 3.2 Opt-in live test

The live tests are gated behind `RISALIVE=1` (they skip by default so they
never block normal development/CI — see `AGENTS.md`):

```powershell
$env:RISALIVE="1"
pytest -q tests/test_ollama_live.py
$env:RISALIVE=""
```

Expected: **4 passed** (reachability/version, model-presence, tiny chat,
streaming chat).

> `OLLAMA_MODEL` defaults to `qwen3-coder:30b`. If you want a different installed
> model for the tests, set `OLLAMA_MODEL` in the same shell.
>
> `test_live_ollama_tiny_chat_non_empty` is a **non-streaming** chat with a
> **180 s** client-timeout ceiling (raised from 30 s to accommodate slow,
> CPU-bound generation on multi-GB local models; generation is token-rate
> limited, not a failure). `chat_stream` is exercised by the other live tests,
> but the agent loop's primary request path is non-streaming
> `chat_resilient`.

### 3.3 Configuration knobs

The default model is `qwen3-coder:30b`, overridable per run via env or CLI flags.
Generation parameters flow **only** to the native Ollama `/api/chat` `options`
(never to an OpenAI-compatible endpoint).

| Env var | CLI flag | Default | Meaning |
|---|---|---|---|
| `OLLAMA_BASE_URL` | `--base-url` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `--model` | `qwen3-coder:30b` | Model id |
| `AGENT_KEEP_ALIVE` | `--keep-alive` | `30m` | Keep the model resident, e.g. `30m` |
| `AGENT_REQUEST_TIMEOUT` | `--request-timeout` | `600` | Per-request budget (s) |
| `AGENT_NUM_CTX` | `--num-ctx` | `65536` | Context window size (max-chunking: 8192/chunk, 300k via shards) |
| `AGENT_NUM_PREDICT` | `--num-predict` | `16384` | Max tokens generated |
| `AGENT_MAX_RETRIES` | `--max-retries` | `2` | Retries past 1st attempt |
| `AGENT_BACKOFF_S` | `--backoff-s` | `2.0` | Base backoff between retries |
| `AGENT_EXPERIENCE_ENABLED` | — | `true` | Learn from verified runs (see README) |
| `AGENT_EXPERIENCE_PATH` | — | `~/.risa/ascs/` | Where the experience JSONL store lives |

Example:

```powershell
risa --model qwen3-coder:30b --num-ctx 16384 --auto "Add a --verbose flag"
```

---

## 4. OpenCode setup and validation

The ASCS repo already ships an `opencode.json` that exposes the local model to
OpenCode using the OpenAI-compatible adapter (no API key needed):

```jsonc
"provider": {
  "ollama": {
    "npm": "@ai-sdk/openai-compatible",
    "name": "Ollama (local)",
    "options": { "baseURL": "http://localhost:11434/v1" },
    "models": {
      "qwen3-coder:30b": {
        "name": "Qwen3-Coder 30B (local)",
        "limit": { "context": 65536, "output": 16384 }
      },
      "qwen2.5-coder:14b": {
        "name": "Qwen2.5-Coder 14B (fallback, 16GB)",
        "limit": { "context": 32768, "output": 8192 }
      }
    }
  }
}
```

### 4.1 Confirm the model is visible

Run OpenCode from the repo root, then open the model picker:

```powershell
cd C:\path\to\ASCS
opencode
# in the TUI:  /models  ->  look for  ollama/qwen3-coder:30b
```

### 4.2 Smoke probes

1. **Deterministic probe** — prompt: `Reply with exactly: OPENCODE LOCAL QWEN TEST OK`.
   Expect the exact string back (proves the `/v1` route + model work end to end).
2. **Tool-call probe** — ask something that requires the local filesystem, e.g.
   `List the files in this repo's docs/ directory.` This confirms OpenCode can
   run tools and reach the same local server reliably.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` on `:11434` | Ollama service not running | Start Ollama; verify `curl.exe http://localhost:11434/api/version` |
| `Model 'qwen3-coder:30b' NOT INSTALLED` in `risa` | Model not pulled | `ollama pull qwen3-coder:30b`; confirm via `/api/tags` |
| OpenCode shows no `ollama` provider / no model | `/v1/models` empty on the server | Confirm via `curl.exe http://localhost:11434/v1/models`; pull the model |
| 404 when requesting the model | model id mismatch | Compare exact id (`qwen3-coder:30b` vs `qwen3-coder:30b:latest`) in `/v1/models`; align `opencode.json` / `OLLAMA_MODEL` |
| Very slow / out-of-memory | too much RAM for 30B (19GB Q4) | Use ≥32 GB (fallback `qwen2.5-coder:14b` on 16GB: `risa --model qwen2.5-coder:14b`); lower `n.ctx` via `AGENT_NUM_CTX` (e.g. 32768) and `opencode.json` `limit.context` |
| OpenCode can't find `@ai-sdk/openai-compatible` | npm package not fetchable / node missing | Install Node.js; retry OpenCode startup |

---

## 6. Acceptance checklist

- [ ] `curl.exe http://localhost:11434/api/version` succeeds
- [ ] `curl.exe http://localhost:11434/v1/models` lists `qwen3-coder:30b`
- [ ] `curl.exe http://localhost:11434/api/tags` lists `qwen3-coder:30b`
- [ ] `risa --check` reports reachable and available
- [ ] `risa --list-models` shows `qwen3-coder:30b` INSTALLED
- [ ] `RISALIVE=1 pytest -q tests/test_ollama_live.py` → 4 passed
- [ ] OpenCode `/models` shows `ollama/qwen3-coder:30b`
- [ ] "OPENCODE LOCAL QWEN TEST OK" probe returns exactly
- [ ] Repo-inspection tool-call probe succeeds
