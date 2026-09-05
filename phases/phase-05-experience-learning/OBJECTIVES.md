# Phase 5 — Experience-Based Learning — Objectives

Build bounded, persistent learning on top of the existing experience store
(`agent/experience/store.py`, JSONL at `~/.risa/ascs/experiences.jsonl`):

- Persistent experience: task, outcome, ordered plan, actions, observations,
  verification result, `success`/`score`.
- Language/toolchain/error-aware retrieval: past wins relevant to the current
  stack and failure shape rank highest.
- Strategy retrieval: inject only a short, bounded `PAST EXPERIENCE` block
  (top few matches, a few KB) into the system/planner prompts.
- Outcome weighting: successes reinforce; runs that contradict a prior
  success penalize it (score floor −1.0) so the model stops trusting stale
  approaches.
- Bounded learning: capped store (5000 records), corrupt-line skipping,
  never blocks a run; CLI on by default (`AGENT_EXPERIENCE_ENABLED`), library
  opt-in, path overridable (`AGENT_EXPERIENCE_PATH`).

Do not turn learning into weight updates or unbounded context growth.
