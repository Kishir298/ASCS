# A.S.C.S. Roadmap — Phases

Source of truth for sequencing: `phases/phase-*/OBJECTIVES.md` (details) and
`ACCEPTANCE.md` (sign-off). Status words: `Done` (Phase 0 on sign-off),
`Planned` (all others — objectives exist, work not started).

```text
Phase 0 — Architecture & Repository Organization ............ Done*
Phase 1 — Brain, Intent & Tool-Use Audit ................... Planned
Phase 2 — True PLAN / BUILD / AUTO Modes ................... Planned
Phase 3 — Windows / PowerShell Execution ................... Planned
Phase 4 — Language Intelligence ............................ Planned
Phase 5 — Experience-Based Learning ........................ Planned
Phase 6 — Verification & Reliability ....................... Planned
```

`*` Phase 0 is `Done` when its `ACCEPTANCE.md` checklist passes (baseline
`7 failed, 578 passed, 5 skipped` preserved, diagnostics green, no
`phases/` imports, diff reviewed).

## What each phase owns

- **Phase 0** — boundaries only: `agent/` subpackages, `phases/` shells,
  `tests/` domains, `docs/architecture/`, `scripts/`. No intelligence changes.
- **Phase 1** — audit intent → plan → tools → verification; duplicated
  decision logic; `hello → write_file` root cause + fix plan.
- **Phase 2** — true PLAN (read-only) / BUILD (plan-first) / AUTO, plus SAFE
  overlay and dirty-guard.
- **Phase 3** — Windows/`cmd.exe` execution, structured failures,
  cancellation/timeouts, live command output.
- **Phase 4** — evidence-based language/framework/toolchain awareness and
  language-specific verification/context.
- **Phase 5** — persistent, bounded experience learning with outcome weighting.
- **Phase 6** — verification gates, bounded retries, cascade-cancel, resume,
  regression coverage (live tests stay `RISALIVE=1` opt-in).

## Non-goals

A.S.C.S. stays standalone (no CORE/RESCS/ASIS/TIVISS), Ollama-local
(`qwen3-coder:30b` / `qwen2.5-coder:14b`), terminal-native, with
model-specific chunking (8192 / 4096).
