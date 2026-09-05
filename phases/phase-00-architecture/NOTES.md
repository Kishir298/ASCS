# Phase 0 — Notes

## Baseline (2026-09-05, Windows, `main` @ `941c2c6`)

- Branch `main`, up to date with `origin/main` (`0 ahead/behind`).
- One unstaged local change at start: `deleted: agent/ui/index.html` (377 lines).
  `agent/web.py` already serves a `_fallback_html()` when the file is missing,
  so the deletion does not break runtime or tests. Treated as intentional
  terminal-native legacy cleanup and kept/documented in Phase 0.
- No untracked files, no stash.
- Baseline suite: `7 failed, 578 passed, 5 skipped in ~76s`
  (`pytest --tb=no -p no:cacheprovider`). All 7 failures are pre-existing TUI
  shell-spec assertions (`test_hello_fits_all_tiers` x2, `test_no_queued_fake_status`,
  `test_clear_preserves_config`, `test_compact_widths_keep_63_col_floor`,
  `test_slash_menu_lists_all_three_commands`, `test_bare_slash_shows_menu`).
- Diagnostics: `python -m agent.terminal --check` → exit 0
  (`Ollama: OK`, `Model 'qwen3-coder:30b': available`).
  `python -m agent.terminal --list-models` lists `qwen3-coder:30b`,
  `qwen2.5-coder:14b`, plus local extras.

## Repository before (flat layout)

- `agent/`: 24 flat `.py` files (~10k lines; `tui.py` alone ~3200 lines).
  Entrypoints: `risa` → `agent.terminal:main` (terminal-native, `--ui` aliases
  `--tui`); `python -m agent` → `agent.__main__` → `agent.main:main`;
  `agent.main --ui` still serves `agent.web:serve`; `agent.tui:run_tui` is the
  primary UI and reuses `agent.web:EventHub/TaskRunner`.
- `tests/`: 31 flat files (`conftest.py` + 30 `test_*.py`), ~588 `def test_*`
  (≈542 unique; `test_full_cli.py` duplicates 46 by design). Deterministic
  fakes/mocks by default; live tests gated behind `RISALIVE=1` (5 tests).
- `docs/`: only `OLLAMA_SETUP.md`. No `architecture/` or `phases/` docs.
- No `scripts/` or `phases/` directories.
- Packaging: `pyproject.toml` `include = ["agent*"]` (covers new subpackages),
  `risa = "agent.terminal:main"`, zero runtime deps, `windows-curses` on Win32.

## Web UI audit

- `agent/web.py` (458 lines): stdlib HTTP + SSE server. `GET /` serves
  `agent/ui/index.html` with graceful fallback when missing — deletion-safe.
- `EventHub` / `TaskRunner` in `web.py` are imported by `agent/tui.py`
  (lines ~813/845/882), so `web.py` cannot be deleted in Phase 0 despite HTTP
  serving being legacy. Documented as transitional shared infrastructure.
- `agent/main.py --ui` path retained for `python -m agent --ui`; terminal entry
  (`agent.terminal`) normalizes `--ui` → `--tui`. No contradictory architecture
  after documenting: terminal-native primary, browser serving legacy/fallback.

## Migration approach (safe movement)

- Pure docs/phases/scripts additions first (zero runtime risk).
- Canonical implementations **moved** into `agent/` subpackages with package
  `__init__.py` files that document ownership (lazy PEP 562 re-exports where
  the old import path equals the package name, e.g. `agent.tools`,
  `agent.context`, `agent.experience`, `agent.models`, `agent.terminal`).
- Thin compatibility shims at old flat paths (`agent/loop.py`,
  `agent/planner.py`, `agent/ollama.py`, `agent/tui.py`, …) re-export the
  canonical modules so both old and new import paths work.
- Where a file and package would share a name, the canonical file was moved
  into the package (e.g. `agent/tools.py` → `agent/tools/core.py`) and the
  old module path is served by lazy re-exports from the package `__init__` —
  no duplicate logic, no breakage, no circular imports.
- `tests/` reorganized by domain with compatibility shims at old paths so
  history/CI references keep working; pytest discovers both old and new paths
  (duplicates avoided by moving, not copying, wherever possible).
- Post-migration suite (2026-09-05): `7 failed, 578 passed, 5 skipped` —
  identical to baseline; all 7 failures pre-existing TUI shell-spec
  assertions, zero migration regressions. `tests/verification/` gained a
  3-test contract suite for the `agent.verification` re-export surface.
