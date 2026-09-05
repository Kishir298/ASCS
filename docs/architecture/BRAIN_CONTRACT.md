# A.S.C.S. Brain Contract (Phase 1)

The authoritative decision architecture: how a user request becomes — or is
refused as — authorized work. Implemented in `agent/core/intent.py` (decision
layer) and `agent/core/loop.py` (orchestration). The model provides
intelligence; the application provides boundaries.

## Request categories

| Category | Meaning | Workspace context | Tools authorized | Writes/commands |
| --- | --- | --- | --- | --- |
| `conversation` | greetings, thanks, small talk, "why did you do that" | never loaded | none | never |
| `question` | general-knowledge / explanation ("what is Python?") | never loaded | none | never |
| `project_inspection` | questions about *this* repository | yes (read-only) | read-only tools | never |
| `code_change` | explicit coding work orders | yes | mode-gated full set | allowed by mode |
| `file_operation` | delete/rename/move a named file | yes | mode-gated full set | allowed by mode |
| `command_request` | "run the tests" | yes | mode-gated full set | allowed by mode |
| `verification_request` | verify reported work | yes | read-only + commands | allowed by mode |
| `ambiguous` | terse work orders ("do the thing", "make a.txt") | yes | mode-gated full set | model must justify per prompt contract |

Classification is **deterministic, pure, and high-confidence-only**: only
greetings/thanks/small-talk, world-knowledge questions, and clearly
project-directed questions are asserted with `high` confidence. Everything
else is `ambiguous`/`moderate`, so terse-but-legitimate work orders keep
working under normal mode gating.

## Decision flow

```text
input
→ classify_request()            (agent/core/intent.py — before any I/O)
→ conversational?  → answer in ONE model turn, zero tools, done
→ else             → load context only if requires_workspace (demand-driven)
                   → model decides tools inside mode gating
                   → mutating tool call + read-only intent?  → refused (max 2)
                   → repeated violation?                     → session ends (fatal)
→ planner only when work is decomposable (run_graph / --tasks)
→ executor runs authorized tasks only
→ verification distinguishes tool success from objective success
→ stop when done / cancelled / retry budget exhausted
```

## Ownership (one owner per decision)

| Decision | Owner |
| --- | --- |
| Run lifecycle, high-level orchestration | `AgentLoop` (`agent/core/loop.py`) |
| Request classification / action necessity | `agent/core/intent.py` |
| Decomposition into structured tasks | `agent/planning/planner.py` |
| Task dependencies and state | `agent/execution/tasks.py` |
| Execution of authorized tasks | `agent/execution/executor.py` |
| Tool validation + invocation | `agent/tools/core.py` |
| Repository knowledge / retrieval | `agent/context/` |
| Bounded evidence for prompts | `agent/experience/store.py` |
| Truth about task success | `agent/execution/executor.py` verification gate |
| Lifecycle state | `agent/core/state.py` |
| What actually happened | `agent/events.py` |

## Side-effect rules (minimum-action principle)

- `hello` must create nothing. A hostile model that always calls `write_file`
  is unable to produce a single file change on conversational requests —
  enforced by the loop, not by prompt obedience (see
  `tests/core/test_intent_side_effects.py`).
- Inspection requests may read, never write.
- Coding requests touch only what the request implies; unrelated files are a
  contract violation.
- Experience influences strategy only; it never fires for requests that
  authorize no workspace work (`requires_workspace=False` skips retrieval and
  recording), and never triggers an action by itself.
- ASCS's own `.ascs/` state directory is internal bookkeeping, not a
  model-visible side effect.

## Recovery rules

- Malformed model replies: bounded by `AGENT_MALFORMED_RETRY_LIMIT` (5) → `FAILED`.
- Intent-gate refusals: one feedback turn; a second violation ends the session
  as `fatal` (no infinite refusal loops).
- Identical consecutive *failing* tool calls: stopped after 2 repeats.
- Verification failures: bounded retries (`AGENT_MAX_VERIFY_RETRIES=2`),
  then task `FAILED` with cascade-cancel of dependents.
- Model timeouts/connection loss: immediate truthful `fatal`.
- Cancellation always reports `cancelled` — never a fake success.

## Stop conditions

Stop when the conversational answer is delivered, the objective is verified,
the operator cancels, work cannot safely continue, or the retry/recovery
budget is exhausted. Remaining iteration budget is never a reason to keep
calling the model.
