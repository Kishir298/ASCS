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

# Diagnostics (read-only PASS/WARN/FAIL report)
risa --doctor

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
- Multi-line input: Enter sends, Shift+Enter inserts a new line
- Live command output: every `run_command` streams its result (stdout,
  stderr, exit code) into the feed so you see exactly what the agent saw
- **STOP** cancels the active run immediately (interrupts the model call,
  kills child processes, reports `cancelled`)
- History, status (Ollama reachable? model installed?), clear

Server-Sent Events at `/api/events`, plus `/api/task`, `/api/stop`,
`/api/state`, `/api/status`, `/api/history`, `/api/clear`.

## Events

The loop emits structured, JSON-serializable events over SSE for the UI and
for future ASIS/TIVISS integrations:

`agent_started`, `status`, `mode_changed`, `model_started`, `model_completed`,
`activity`, `tool_started`, `tool_completed`, `file_read`, `file_written`,
`patch_applied`, `command_started`, `command_output`, `command_completed`,
`test_started`, `test_completed`, `agent_error`, `agent_stopped`,
`agent_completed`.

`command_output` carries the truncated stdout/stderr text of a
`run_command`; a timed-out command is reported with exit code `-1` and never
counts as success in the UI, the events, or the model's view of the result.

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

## Diagnostics (`risa doctor`)

```
risa --doctor [--workspace <path>]
```

Runs a read-only diagnostic suite and reports a `PASS` / `WARN` / `FAIL`
status for each item, with actionable recovery hints:

- Python environment and A.S.C.S. install
- Configuration validity
- Workspace existence and containment
- Ollama connectivity and the configured model
- Registered tools
- Persistent context index health
- Project manifest / discovery state
- Git availability
- pytest availability

## Project intelligence

A.S.C.S. treats the repository as a persistent project, not a sequence of
one-off conversations. While working it builds and maintains:

- **Project manifest** (`.ascs/project_manifest.json`) — languages,
  frameworks, package managers, dependencies, entry points, tests, config and
  documentation files, plus git state.
- **Persistent file index** (`.ascs/context_index.json`) — per-file metadata,
  symbols, imports and dependencies. Repeated sessions reuse the index; the
  index is updated **incrementally** (only changed/new/deleted files are
  re-processed).
- **Hierarchical retrieval** for each task:
  - Level 1: project metadata
  - Level 2: directory/file summaries
  - Level 3: relevant source files, chunked to a token budget
  - Level 4: exact code regions plus dependency- and test-related files

The agent receives a scan-derived "PROJECT INTELLIGENCE" block in its system
prompt, so it starts every task knowing the project's shape instead of
rediscovering it.

## Task engine

Plans and work items are structured as a dependency-aware **task graph**
(`agent.tasks`) with explicit statuses
(`pending → ready → running → completed / failed / blocked / cancelled /
skipped`). Each task carries its own description, dependencies, files,
commands, verification list, retry count and failure reason. Task graphs
persist to `.ascs/task_state.json`, which is the foundation for resuming
interrupted objectives rather than restarting them.

## Configuration

Environment variables (CLI flags win, both override defaults):

| Variable                 | Meaning                              | Default               |
| ------------------------ | ------------------------------------ | --------------------- |
| `AGENT_MODE`             | `PLAN` / `BUILD` / `AUTO` / `SAFE`   | `AUTO`                |
| `AGENT_APPROVAL`         | in SAFE mode, auto-approve all       | off (ask each time)   |
| `AGENT_UI_HOST` / `AGENT_UI_PORT` | web UI bind address / port | `127.0.0.1` / `8787` |
| `OLLAMA_BASE_URL`        | Ollama server URL                    | `http://localhost:11434` |
| `OLLAMA_MODEL`           | Ollama model                         | `qwen2.5-coder:14b`   |
| `AGENT_MAX_ITERATIONS`   | agent iteration budget (→ `TIMEOUT`) | `50`                  |
| `AGENT_REQUEST_TIMEOUT`  | per-model-call timeout (s)           | `600`                 |
| `AGENT_COMMAND_TIMEOUT`  | default `run_command` timeout (s)    | `120`                 |
| `AGENT_KEEP_ALIVE`       | Ollama keep-alive (e.g. `30m`)       | default (Ollama-side) |
| `AGENT_PREWARM`          | warm the model at startup            | on                    |
| `AGENT_VERBOSE`          | show tool output in CLI logs         | off                   |
| `AGENT_MAX_OUTPUT_CHARS` | per-tool-output truncation limit     | `20000`               |
| `AGENT_CONTEXT_BUDGET_CHARS` | rolling conversation budget      | `70000`               |
| `AGENT_MALFORMED_RETRY_LIMIT` | bad-reply retries before `FAILED` | `5`               |

## Development

```bash
pip install -e ".[dev]"
pytest
```

Test suite covers the tool layer, the loop's lifecycle/plan/mode/cancellation
behaviour (using scripted fake clients — no Ollama required), event emission,
the Ollama HTTP client against an in-process mock server, the state machine,
the staged boot, and the web server endpoints.