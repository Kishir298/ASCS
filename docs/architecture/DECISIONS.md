# A.S.C.S. Decisions (Phase 0)

## D1 — Facades over moves where renames would break imports

`agent/tools.py` vs `agent/tools/`, `agent/context.py` vs `agent/context/`,
`agent/experience.py`, `agent/terminal.py`, `agent/models.py` vs
`agent/models/` cannot coexist as file + package. Per the safe-movement rule
we kept canonical files in place and made the domain packages documented
re-export facades. Rationale: ~30 test files and all internal relative
imports use `from agent.<module> import …` / `from .<module> import …`; a
blind move would break the suite and risk circular imports for zero behavior
gain. Future phases may perform true moves file-by-file with shims.

## D2 — Browser UI is legacy, but `web.py` stays

`agent/ui/index.html` is deleted locally and `--ui` aliases to `--tui`, so
browser serving is legacy. But `agent/tui.py` imports `EventHub`/`TaskRunner`
from `agent/web.py`, and `web.py` degrades gracefully (`_fallback_html()`).
Deleting `web.py` in Phase 0 would break the TUI for no benefit. Decision:
document HTTP serving as legacy/transitional, keep `web.py` as shared
infrastructure, revisit in a later phase if the TUI stops depending on it.

## D3 — `phases/` is never runtime

`phases/phase-*/OBJECTIVES.md|ACCEPTANCE.md|NOTES.md` are roadmap artifacts.
Enforced by `scripts/verify_phase0.py` (grep for `from phases` / `import
phases` under `agent/`, `tests/`, `scripts/`). Failed check blocks Phase 0
sign-off.

## D4 — Tests keep old import paths working

Test files use absolute `from agent.<module> import …`. Domain folders under
`tests/` (`core`, `planning`, `execution`, `tools`, `context`, `experience`,
`verification`, `terminal`, `integration`) organize by what is actually
tested; thin compatibility modules at the old flat paths preserve history/CI
references. Pytest discovers both; no test logic was changed in Phase 0.

## D5 — Locked parameters are non-negotiable

Standalone A.S.C.S.; Ollama-local; `qwen3-coder:30b` + `qwen2.5-coder:14b`
only; model-specific chunking 8192/4096; terminal-native. Any proposal
touching these in Phases 1–6 must be rejected or re-scoped.

## D6 — No Phase 1 work in Phase 0

The `hello → write_file` symptom and any brain/mode/shell/language/learning/
reliability redesign are explicitly deferred. Phase 0 creates the address
(`agent.core`, `agent.planning`, …) and the test/docs structure so Phase 1 can
attack the problem with a clean baseline.
