# Phase 0 — Architecture & Repository Organization — Objectives

Establish a clean, scalable repository and runtime architecture for A.S.C.S.
without changing agent intelligence or behavior.

Scope (this phase only):

- Inspect the existing repository (`agent/`, `tests/`, `docs/`, top-level files)
  and follow actual imports and runtime call paths (do not trust filenames).
- Inspect git state (branch, log, uncommitted work) without resetting or
  discarding user work.
- Separate concerns: `agent/` = runtime only, `phases/` = dev artifacts only,
  `tests/` = automated tests, `docs/` = documentation, `scripts/` = maintenance.
- Create `agent/` subpackage boundaries: `core`, `planning`, `execution`,
  `tools`, `context`, `experience`, `verification`, `models`, `terminal`.
  Move code where safe; where a move would break imports, keep the canonical
  file in place and document the boundary (per safe-movement rule).
- Create `phases/phase-00` … `phase-06` roadmap shells (objectives/acceptance/
  notes). `phases/` must never be imported by runtime code.
- Create `docs/architecture/` (actual post-Phase-0 code, not aspirations) and
  `docs/phases/ROADMAP.md`.
- Organize `tests/` by runtime domain while keeping pytest discovery working.
- Audit the browser/web UI (`agent/web.py`, deleted `agent/ui/index.html`,
  `--ui` flag): `web.py` `EventHub`/`TaskRunner` are still reused by the TUI;
  HTTP serving is legacy. Terminal/TUI (`agent/tui.py` + `agent/terminal.py`)
  is primary. Document, do not leave contradictory architecture.
- Preserve locked parameters: standalone A.S.C.S. (no CORE/RESCS/ASIS/TIVISS),
  Ollama-local, `qwen3-coder:30b` primary + `qwen2.5-coder:14b` fallback (never
  `qwen3-coder:14b`), model-specific chunking 8192 / 4096.

Explicitly out of scope: Phase 1 brain audit (including any `hello →
write_file` fix), Phase 2 modes, Phase 3 PowerShell, Phase 4 language intel,
Phase 5 learning, Phase 6 reliability.
