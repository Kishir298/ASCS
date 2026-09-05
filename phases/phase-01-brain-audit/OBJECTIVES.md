# Phase 1 — Brain, Intent & Tool-Use Audit — Objectives

Audit (do not yet redesign beyond what the audit requires) the agent's
decision-making core:

- Intent handling: how a raw user task becomes an actionable objective.
- `AgentLoop` (`agent/core/loop.py`): single-shot loop, iteration
  budget, malformed-reply handling, context-budget trimming, cancellation.
- Planner (`agent/planner.py`, `agent/planning/`): objective → validated DAG,
  large-task chunking, verification guarantees.
- Executor (`agent/executor.py`, `agent/execution/`): task-scoped prompts,
  verification gate (`run …` must exit 0), bounded retries, cascade-cancel.
- Tool execution (`agent/tools/core.py`): validation, containment,
  truncation, timeouts, git dirty-guard.
- Prompts (`agent/prompts.py`): system/task/malformed/tool-error prompts per
  mode; `PROJECT INTELLIGENCE` + `PAST EXPERIENCE` injection.
- Project index (`agent/context/index.py`, `agent/context/project.py`):
  manifest, incremental index, hierarchical L1–L4 retrieval, model-specific
  chunking (30b: 8192 / 14b: 4096).
- Duplicated decision logic across loop/planner/executor/prompts; propose a
  unified planning → execution → recovery flow.
- Known symptom to investigate (not fix in Phase 0): trivial intents such as
  `hello` reaching `write_file`. Record root cause and a minimal, tested fix
  plan for Phase 1 execution.

Deliverable: findings + a focused refactor plan with regression tests. No
Phase 2–6 scope creep.
