# A.S.C.S. — A Smart Coding System

A local, private, autonomous coding agent built on **Ollama**. It plans,
writes, runs, and verifies code inside an explicit workspace directory, and
keeps humans in control through three session modes, a live web UI, and
real STOP/cancellation.

- **Zero runtime dependencies** — standard library only, driven by your local
  Ollama server. No code ever leaves your machine.
- **Explicit lifecycle** — every run moves through
  `RECEIVING_TASK → PLANNING → EXECUTING → VERIFYING → COMPLETE`
  (or `FAILED` / `CANCELLED` / `TIMEOUT`), visible live in the UI.
- **Structured plans** — the agent records an ordered plan with `set_plan`;
  in BUILD mode it must plan before it edits, and the plan is shown to you.
- **Windows-aware shell** — spawned command runs resolve `python`/`pip`/
  `pytest` and follow real `cmd.exe` workflows instead of Unix habits.
- **Real cancellation** — STOP interrupts the model call, kills child
  processes, and reports `cancelled` (never a false success).

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) running locally (`ollama serve`, default
  `http://localhost:11434`)
- A code model, e.g. `ollama pull qwen2.5-coder:14b`

## Install

```bash
pip install -e .
risa --check
```

The default model is `qwen2.5-coder:14b`. Override it per run with
`--model`, or set any model via environment variables (see Configuration).

## Quick start

```bash
# One-shot autonomous run
risa --auto "Add a --verbose flag to the CLI and test it"

# Plan first (read-only: inspect + record a plan, no edits)
risa --mode plan "How should we split the config loader into modules?"

# Build mode: plan, get the plan shown, then implement
risa --mode build --workspace ./backend "Refactor the auth module"

# Web UI (opens at http://127.0.0.1:8787)
risa --ui
```

The CLI automatically runs a real startup sequence (config → environment →
workspace → Ollama connectivity → model availability → tools) and reports each
stage's timing and any recovery hint on failure.

## Modes

| Mode    | Behaviour                                                        |
| ------- | ---------------------------------------------------------------- |
| `PLAN`  | Read-only: lists, reads, searches, `inspect_environment`, git, `set_plan`. No writes, no commands. |
| `BUILD` | Records an approved plan first, then implements and tests it.    |
| `AUTO`  | Fully autonomous end-to-end run.                                 |
| `SAFE`  | Legacy overlay: every modifying action asks for approval (y/N).  |

`--safe` and `--auto` conflict; PLAN/BUILD/AUTO are set with `--mode`.

## Web UI

```
risa --ui [--host 127.0.0.1] [--port 8787]
```

The default `--host 127.0.0.1` binds to localhost only; use `0.0.0.0` if you
want other machines on your LAN to reach it.

- Chat area with live state pill and per-step output
- Mode selector (PLAN / BUILD / AUTO) per task
- **STOP** cancels the active run immediately (interrupts the model call,
  kills child processes, reports `cancelled`)
- History, status (Ollama reachable? model installed?), clear

Server-Sent Events at `/api/events`, plus `/api/task`, `/api/stop`,
`/api/state`, `/api/status`, `/api/history`, `/api/clear`.

## Tools

`list_directory`, `read_file`, `search_files`, `write_file`, `apply_patch`,
`delete_file`, `move_file`, `copy_file`, `run_command`, `inspect_environment`,
`git_status`, `git_diff`, `set_plan`.

Hard guarantees:

- Everything resolves inside the workspace root; `..` / absolute / symlink
  escapes are rejected (`WorkspaceError`).
- `delete_file` refuses the root, directories, and VCS metadata
  (`.git`, `.github`, `.gitignore`).
- PLAN mode disables all mutating tools; SAFE gates them behind approval.

## Configuration

Environment variables (CLI flags win, both override defaults):

| Variable                 | Meaning                              | Default               |
| ------------------------ | ------------------------------------ | --------------------- |
| `AGENT_MODE`             | `PLAN` / `BUILD` / `AUTO` / `SAFE`   | `AUTO`                |
| `AGENT_APPROVAL`         | in SAFE mode, auto-approve all       | off (ask each time)   |
| `AGENT_UI_HOST` / `AGENT_UI_PORT` | web UI bind address / port | `127.0.0.1` / `8787` |
| `AGENT_MODEL`            | Ollama model                         | `qwen2.5-coder:14b`   |
| `AGENT_REQUEST_TIMEOUT`  | per-model-call timeout (s)           | `600`                 |
| `AGENT_KEEP_ALIVE`       | Ollama keep-alive (e.g. `30m`)       | default (Ollama-side) |
| `AGENT_PREWARM`          | warm the model at startup            | on                    |

## Development

```bash
pip install -e ".[dev]"
pytest
```

Test suite covers the tool layer, the loop's lifecycle/plan/mode/cancellation
behaviour (using scripted fake clients — no Ollama required), event emission,
the Ollama HTTP client against an in-process mock server, the state machine,
the staged boot, and the web server endpoints.