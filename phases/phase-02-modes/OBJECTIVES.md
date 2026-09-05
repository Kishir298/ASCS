# Phase 2 — True PLAN / BUILD / AUTO Modes — Objectives

Make the three session modes genuinely distinct (plus the `SAFE` approval
overlay):

- `PLAN`: read-only — inspect, read, search, `inspect_environment`, git
  inspection, `set_plan`. No writes, no commands, no verification commands.
- `BUILD`: must record an approved plan first, then implement and test it.
  Git dirty-state guard protects pre-existing user work.
- `AUTO`: fully autonomous end-to-end run within iteration/timeout budgets.
- `SAFE`: legacy overlay — every modifying action and verification command
  requires operator approval (y/N).

Scope: mode gating in the loop, the task engine, and the tool layer; mode
visibility in the terminal/TUI and events. Do not redesign the brain, shell,
language intel, learning, or verification internals beyond what mode gating
requires.
