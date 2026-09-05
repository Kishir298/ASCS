# Phase 1 — Acceptance Criteria

- [x] Intent pipeline mapped end-to-end (task → plan → tools → verification).
- [x] `AgentLoop`, planner, executor, tool execution, prompts, and project
  index responsibilities documented (see `docs/architecture/BRAIN_CONTRACT.md`).
- [x] Duplicated planning/execution/recovery decision logic listed with
  unification proposal (planner fallback unified via `fallback_spec_for`).
- [x] Root cause of trivial-intent misrouting (`hello → write_file`)
  identified and fixed via the intent/decision layer
  (`agent/core/intent.py`) + loop gate; regression tests in
  `tests/core/test_intent.py` and `tests/core/test_intent_side_effects.py`.
- [x] No mode, PowerShell, language-intel, learning, or reliability redesigns
  performed as part of the audit.
- [x] Findings recorded in `phases/phase-01-brain-audit/NOTES.md`.
