# Phase 1 — Notes

Status: `Implemented` (2026-09-05) — brain audit + intent-aware orchestration
landed. Phase 0 completed first (see `phases/phase-00-architecture/NOTES.md`).

## Root cause of `hello → write_file`

Traced through the Phase 0 code (`agent/core/loop.py`):

1. `AgentLoop.run()` loaded the FULL system prompt for every request —
   including the AUTO-mode instruction *"plan, inspect, implement, test,
   debug, and verify until the task is genuinely done"* — plus the complete
   PROJECT INTELLIGENCE block (eagerly built by scanning the repo) and any
   PAST EXPERIENCE entries whose text claimed success on similar input.
2. Nothing in the pipeline distinguished a conversational message from a work
   order. The response contract offered 13 tools, and the prompt explicitly
   told the model to *"record a short plan with set_plan early"*.
3. A small local model, primed with coding-agent framing + project
   intelligence + experience entries, picked the most coding-agent-shaped
   action for `hello`: `write_file hello.txt`, then read/verified it.
4. Secondary amplifier: the planner's failure fallback (`loop.py
   _plan_objective`, `planner.py plan_objective`) converted ANY objective —
   including conversation — into an implement-and-verify task, so the
   `--tasks` path guaranteed coding work even for greetings.

Causal chain: eager context → coding-agent persona → tool contract →
no intent boundary → `write_file`. The application had NO boundary; the
model's behavior was the only gate.

## New decision architecture

- `agent/core/intent.py` (new): deterministic, pure, high-confidence-only
  classifier producing a `Decision` (intent, confidence, `requires_*` flags,
  scope, reason). Categories: conversation, question, project_inspection,
  code_change, file_operation, command_request, verification_request,
  ambiguous. Only social input, world-knowledge questions, and clearly
  project-directed questions are `high` confidence — terse-but-legitimate work
  orders (`do the thing`, `make a.txt`) stay `ambiguous` and executable.
- `agent/core/loop.py`: classification happens BEFORE any orchestration
  decision. Conversational requests skip project indexing and experience
  retrieval entirely (demand-driven context), receive an explicit
  no-side-effect instruction, and are answered in ONE model turn with zero
  tools. A unified intent gate refuses mutating tools on
  conversation/question/project_inspection intents (feedback once, then a
  bounded `fatal` stop on a second violation — no infinite refusal loops).
  The planner fallback is intent-aware (`fallback_spec_for`): planner failure
  on conversational objectives yields a review task, never implementation.
  Experience retrieval/recording is skipped for `requires_workspace=False`
  requests — experience informs strategy, never triggers actions.
- `docs/architecture/BRAIN_CONTRACT.md` documents categories, flow,
  ownership, side-effect rules, recovery, and stop conditions.

## Test evidence

- `tests/core/test_intent.py`: 63 unit tests (categories, flags, contract
  surface, determinism, fallback intent-awareness).
- `tests/core/test_intent_side_effects.py`: 17 loop-level tests with a
  *hostile* client that always calls `write_file` — proves `hello`, `hi`,
  `thanks`, `what can you do?`, `what is Python?`, and inspection requests
  produce zero file changes even against a non-compliant model; pins the
  bounded repeated-violation stop; pins that terse work orders still execute.
- Full suite after Phase 1: `7 failed, 661 passed, 5 skipped` — the 7
  failures are the pre-existing TUI shell-spec assertions from the Phase 0
  baseline (`7 failed, 581 passed, 5 skipped`); zero regressions, +80 tests.

## Duplicated decision logic found

- Planner fallback existed in TWO places (`loop._plan_objective`,
  `planner.plan_objective`) — now both delegate to `fallback_spec_for`.
- Verification derivation was already shared (`_derive_verification`); loop
  and planner both call it. Left as-is.
- Mode gating (PLAN/SAFE tool filtering) lives in config + executor + loop —
  consolidated enough for Phase 1; further unification deferred.
