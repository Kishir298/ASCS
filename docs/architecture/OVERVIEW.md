# A.S.C.S. Architecture — Overview

A.S.C.S. (A Smart Coding System) is a standalone, local, autonomous coding
agent backed by Ollama. It plans, writes, runs, and verifies code inside an
explicit workspace directory. No code leaves the machine; the only runtime
dependencies are the standard library plus `windows-curses` on Windows.

## Locked parameters

- Standalone subsystem: no CORE / RESCS / ASIS / TIVISS dependency.
- Provider: Ollama-local. Primary `qwen3-coder:30b`, fallback
  `qwen2.5-coder:14b` (never `qwen3-coder:14b`).
- Chunking is model-specific: 30b → 8192 tokens, 14b → 4096 tokens
  (`agent/context/index.py`, dual limits). No global chunk size.
- Runtime is Windows-only (`risa --tui`, Ollama); dev testing (`pytest`) is
  cross-platform via the `python → python3` fallback in
  `agent/tools/core.py`.
- Terminal-native: `risa` defaults to the curses TUI. `--ui` is a compat alias
  for `--tui`. Browser serving (`agent/web.py` HTTP/SSE + deleted
  `agent/ui/index.html`) is legacy; `EventHub`/`TaskRunner` in `web.py` remain
  shared infrastructure reused by the TUI.

## Repository separation

- `agent/` — runtime implementation only (canonical code lives in domain
  subpackages; thin shims at the old flat import paths; see `RUNTIME.md`).
- `phases/` — development artifacts only (objectives/acceptance/notes per
  phase). Never imported by runtime code.
- `tests/` — automated tests, organized by runtime domain + `integration/`.
- `docs/` — architecture and roadmap (`architecture/`, `phases/ROADMAP.md`,
  plus `OLLAMA_SETUP.md` for machine setup).
- `scripts/` — maintenance helpers (e.g. `verify_phase0.py`).

## Request flow (actual)

```text
USER → Terminal/TUI → AgentLoop → Intent gate (classify_request) →
  [conversational → one-turn answer, zero tools] →
  Context + Experience (demand-driven) + Config → Planner (conditional) →
  Task/DAG → Executor → Tools → Verification → Recovery/Result → AgentLoop
```

The classic single-shot loop (`risa "…"`) is the default; the task-graph
engine (`risa --tasks "…"`) is CLI-only and not wired into the browser UI.
See `RUNTIME.md` for the boundary map and `COMPONENTS.md` for file ownership.
