# Phase 3 — Acceptance Criteria

- [ ] `run_command` follows Windows/`cmd.exe` semantics with correct
  `python`/`pip`/`pytest` resolution.
- [ ] Timeouts, truncation, and exit codes are structured and truthful
  (timeout never counts as success).
- [ ] Cancellation kills model calls and child-process trees and reports
  `cancelled`.
- [ ] Command output streams live into the terminal/TUI and events.
- [ ] Cross-platform dev (`pytest` on macOS/Linux) keeps working via the
  existing `python → python3` fallback.
