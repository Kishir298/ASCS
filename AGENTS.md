# AGENTS.md

## Test execution rule (IMPORTANT)

When running the test suite, always **skip tests that require the local
`qwen3-coder:30b` (fallback `qwen2.5-coder:14b` on 16GB) Ollama model** or a running live Ollama server.

- The live tests are opt-in and gated behind `RISALIVE=1`. They are correctly
  **skipped by default** (`538 passed, 5 skipped` on the reference Windows
  machine after max-chunking; the exact counts vary with platform/plugins).
- Do **not** set `RISALIVE=1` unless the user explicitly asks for live
  integration testing.
- Confirm `RISALIVE` is unset in the environment before running `pytest`, and
  never run a live test that needs `qwen3-coder:30b` / a running server during
  normal development or CI.

Note: `qwen3-coder:30b` (or fallback `qwen2.5-coder:14b`) may not be installed or the Ollama server may not be running
on a given machine. Treat live-model verification as a separate, explicit,
opt-in step. Max-chunking splits 300k-token objectives into ~8k shards within the 65k window.

## Platform

- **Runtime (risa --ui/--tui, Ollama): Windows-only** (32 GB target). `risa --doctor` warns on non-Windows.
- **Dev testing (pytest): cross-platform.** `agent/tools/core.py:508` transparently maps `python` -> `python3` on POSIX where only `python3` exists, so the suite passes on macOS/Linux too (use `.venv/bin/python -m pytest` or plain `python3 -m pytest`).

## Test commands

- Full deterministic suite: `pytest -q` (expect `564 passed, 6 skipped` after TUI; legacy `538 passed, 5 skipped`)
- Ollama client unit tests: `pytest -q tests/test_ollama.py`
- Experience-store / pipeline tests: `pytest -q tests/test_experience_pipeline.py`
- Live smoke (opt-in, requires running Ollama + `qwen3-coder:30b`):
  - bash: `RISALIVE=1 pytest -q tests/test_ollama_live.py`
  - PowerShell: `$env:RISALIVE="1"; pytest -q tests/test_ollama_live.py`

Live-model note: `test_live_ollama_tiny_chat_non_empty` uses a **non-streaming**
chat with a 180 s client-timeout ceiling (raised from 30 s to accommodate
slow, CPU-bound generation on multi-GB local models; generation is
token-rate limited, not a pipeline failure). `chat_stream` is exercised by the
other live tests and by `OllamaClient`, but the agent loop's primary
request path is non-streaming `chat_resilient`.

For the full from-scratch Windows setup, connection, and verification of the
local `qwen3-coder:30b` (fallback `qwen2.5-coder:14b`) model with OpenCode and the ASCS CLI, see
`docs/OLLAMA_SETUP.md`. Run on a ≥32 GB machine (16GB → fallback), not a low-RAM dev laptop.

## Scratch outputs (Ollama_tests)

- Model-generated scratch/demo/test outputs go in `Ollama_tests/` (gitignored sandbox, never committed).
- Keep real source edits in their normal repo paths; `Ollama_tests/` is for disposable outputs only.
