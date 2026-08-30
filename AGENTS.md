# AGENTS.md

## Test execution rule (IMPORTANT)

When running the test suite, always **skip tests that require the local
`qwen3:14b` Ollama model** or a running live Ollama server.

- The live tests are opt-in and gated behind `RISALIVE=1`. They are correctly
  **skipped by default** (`422 passed, 6 skipped`).
- Do **not** set `RISALIVE=1` unless the user explicitly asks for live
  integration testing.
- Confirm `RISALIVE` is unset in the environment before running `pytest`, and
  never run a live test that needs `qwen3:14b` / a running server during
  normal development or CI.

Note: `qwen3:14b` may not be installed or the Ollama server may not be running
on a given machine. Treat live-model verification as a separate, explicit,
opt-in step.

## Test commands

- Full deterministic suite: `pytest -q` (expect `422 passed, 6 skipped`)
- Ollama client unit tests: `pytest -q tests/test_ollama.py`
- Live smoke (opt-in, requires running Ollama + `qwen3:14b`):
  - bash: `RISALIVE=1 pytest -q tests/test_ollama_live.py`
  - PowerShell: `$env:RISALIVE="1"; pytest -q tests/test_ollama_live.py`

For the full from-scratch Windows setup, connection, and verification of the
local `qwen3:14b` model with OpenCode and the ASCS CLI, see
`docs/OLLAMA_SETUP.md`. Run on a ≥16/32 GB machine, not a low-RAM dev laptop.
