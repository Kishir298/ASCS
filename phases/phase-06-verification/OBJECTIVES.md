# Phase 6 — Verification & Reliability — Objectives

Make every claim of success checkable:

- Verification: each task carries acceptance criteria; every `run …`
  verification step must exit 0; implementing tasks without verification are
  reported as not fully verified, never silently passing.
- Reliability: bounded re-verification (`max_verify_retries=2`,
  `AGENT_MAX_VERIFY_RETRIES`), explicit `verification_started` / `retry` /
  `task_verified` / `task_failed` events with 1-based `attempt` and
  `retries_left`, failure feedback to the model.
- Recovery: task failure cascade-cancels dependents; interrupted runs persist
  to `.ascs/task_state.json` and resume (stuck `RUNNING` resets to `READY`
  without restarting completed work).
- Regression coverage: deterministic fakes/mocks (no Ollama required);
  live tests stay opt-in (`RISALIVE=1`, `qwen3-coder:30b` / `qwen2.5-coder:14b`
  fallback). Timeouts report `TIMEOUT`, cancellations report `cancelled`.
