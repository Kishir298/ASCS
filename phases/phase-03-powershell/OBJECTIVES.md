# Phase 3 — Windows / PowerShell Execution — Objectives

Harden command execution for the Windows-native runtime:

- Windows/`cmd.exe` workflows (no Unix habits): `python`/`pip`/`pytest`
  resolution, PATH interpreter injection, `python → python3` fallback only on
  POSIX dev machines.
- `run_command` semantics: working directory containment, timeouts (default
  `AGENT_COMMAND_TIMEOUT=120s`), truncation (`AGENT_MAX_OUTPUT_CHARS`), exit
  codes (timeout = `-1`, never success), child-process tree kill
  (`taskkill /T`).
- Structured failures: stdout/stderr/exit-code surfaced to the model, the
  events (`command_started`/`command_output`/`command_completed`), and the UI.
- Cancellation: STOP interrupts the model call, aborts the Ollama socket, and
  kills child processes; the run reports `cancelled`, never false success.
- PowerShell-aware improvements only as far as the tool layer requires; no
  fake tool implementations.
