# Phase 2 — Acceptance Criteria

- [ ] `PLAN` runs perform zero writes/commands (verified by tests).
- [ ] `BUILD` runs record a plan before any edit and show it to the operator.
- [ ] `AUTO` runs complete end-to-end without manual steps within budget.
- [ ] `SAFE` gates every modifying tool and verification command on approval.
- [ ] Mode is visible in the terminal/TUI and emitted in events (`mode_changed`).
- [ ] Git dirty-state guard blocks overwriting pre-existing user work in
  BUILD/AUTO unless explicitly approved.
