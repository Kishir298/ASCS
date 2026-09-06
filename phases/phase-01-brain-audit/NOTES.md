# Phase 1 — Notes

Status: `Implemented` (2026-09-05) — brain audit + intent-aware orchestration
landed. Phase 0 completed first (see `phases/phase-00-architecture/NOTES.md`).

Status: `Repaired` — gap closure landed on top of `417725f` (see Gap closure
below): granular file_operation / command_request / verification_request
intents, explicit `run_graph()` question no-work boundary, unified
intent-aware malformed-graph fallback, and the executor intent gate.

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

## Gap closure (repair on top of 417725f)

1. **run_graph no-work boundary** — `run_graph()` now explicitly refuses the
   graph pipeline for high-confidence `conversation` AND `question`
   (`agent/core/loop.py`), answering directly with no refresh, no planner, no
   tools, and truthful `COMPLETE`-only state. Previously only the generic
   `is_conversational` property guarded it and misclassified questions (e.g.
   `what does authentication mean?` → ambiguous) entered full planning.
2. **Granular intents** — `agent/core/intent.py` now genuinely produces all
   eight categories with precedence conversation → file_operation →
   command_request → verification_request → project_inspection → question →
   code_change → ambiguous. `run pytest` is `command_request`, `delete
   foo.py` is `file_operation`, `verify the changes` is
   `verification_request`; `add authentication` / `add tests for the parser`
   stay `code_change`; `what is authentication?` stays `question`.
3. **Unified fallback** — the malformed-graph `except` in
   `loop._plan_objective` no longer hardcodes `Implement and verify`; every
   planner/graph fallback routes through `fallback_spec_for()`, which is now
   per-intent (review / inspect / scoped implement / verify-run /
   conservative review). No fallback can escalate a non-mutating intent.
4. **Executor intent gate** — `TaskExecutor` accepts the run's `Decision` and
   refuses mutating tools for read-only intents inside tasks, mirroring the
   single-shot loop gate (`intent=None` preserves legacy direct-construction
   behavior).
5. **Tests** — `tests/core/test_intent.py` extended (granular positives,
   precedence, question/code boundaries, per-intent fallbacks);
   `tests/core/test_run_graph_intent.py` added (12 tests: run_graph no-work
   incl. questions, planner-skip, malformed recovery per intent, executor
   gate + compat).
