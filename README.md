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

- **Python 3.12+**. ASCS declares `requires-python = ">=3.12"` and otherwise
  uses only the standard library. It has been tested on Python 3.14. On
  Windows, `python` / `py` must resolve to a 3.12+ interpreter. Note: the
  Windows launcher alias `py -3.12` is not necessarily installed; any 3.12+
  runtime satisfies the requirement.
- [Ollama](https://ollama.com) running locally (`ollama serve`, default
  `http://localhost:11434`)
- A code model, e.g. `ollama pull qwen3-coder:30b` (the default model, 30B coder ~19 GB; fallback `ollama pull qwen2.5-coder:14b` for 16 GB)

> **Platform note:** The full runtime (`risa --ui` / `--tui`, Ollama) is
> **Windows-only** (32 GB recommended target). **Dev testing (`pytest`) is
> cross-platform:** on macOS/Linux `python` transparently falls back to
> `python3` at execution time (`agent/tools/core.py:508`), so `pytest` passes
> inside or outside a venv. Use `.venv/bin/python -m pytest -q` on any OS for
> the canonical run.

## Install (Windows PowerShell)

Create a virtual environment and install the project from `requirements.txt`
(which installs the package itself — providing the `risa` entry point — plus
the test/development dependency):

```powershell
cd C:\Users\<you>\Desktop\RISARMS\ASCS

# 1. create a fresh environment with an installed 3.12+ Python
py -m venv .venv

# 2. activate it
.\.venv\Scripts\Activate.ps1

# 3. upgrade packaging tools
python -m pip install --upgrade pip

# 4. install this project and its dependencies
python -m pip install -r requirements.txt

# 5. verify the local environment
risa --doctor
risa --check
```

If you can't activate the environment, call the venv's interpreter directly:
`.\.venv\Scripts\python.exe -m agent --doctor`.

The project is a zero-runtime-dependency package (standard library only),
so a fresh clone needs only `pip install -r requirements.txt` in a valid
venv to run `risa`.

> **No-activation alternative:** the checked-in `risa.cmd` launcher runs the
> venv directly (`.venv\Scripts\python.exe -m agent %*`) and works without
> activating the environment, as long as `.venv` has been created and
> `requirements.txt` installed.

The default model is `qwen3-coder:30b` (fallback `qwen2.5-coder:14b` for 16 GB via `--model`/`OLLAMA_MODEL`). Override per run with `--model`, or set any model via environment variables (see Configuration).

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

# Task-graph engine: split the objective into a DAG of tasks and run them
# task-by-task (each verified), persisted to .ascs for resuming
risa --tasks "Refactor the auth module"

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

> **Limitation:** the web UI runs the single-shot loop
> (`agent.loop.run`). The task-graph engine (`--tasks`) is available from the
> CLI (`risa --tasks "…"`) but is **not** wired into the browser UI; `--ui`
> ignores `--tasks`.

## Events

The loop emits structured, JSON-serializable events over SSE for the UI and
for future ASIS/TIVISS integrations:

`agent_started`, `status`, `mode_changed`, `model_started`, `model_completed`,
`activity`, `tool_started`, `tool_completed`, `file_read`, `file_written`,
`patch_applied`, `command_started`, `command_output`, `command_completed`,
`test_started`, `test_completed`, `agent_error`, `agent_stopped`,
`agent_completed`, `task_plan`, `task_created`, `task_ready`,
`task_started`, `task_blocked`, `task_verified`, `verification_started`,
`task_failed`, `task_completed`, `retry`.

The task-engine run additionally emits `task_plan` (the rendered plan for
operator inspection), `task_created` for every task when the graph is built,
`task_ready` as a task's dependencies are satisfied, `task_started` /
`task_completed` / `task_blocked` for each task as it runs or is blocked,
`task_verified` / `task_failed` at the verification quality gate, and a
human-readable step log (`task_started`/`task_completed` carry the task id
in `status`).

Verification is observable and explicit: `verification_started` announces each
attempt, `retry` (with structured `attempt` and `retries_left` fields) marks a
bounded re-verification, and the final `task_verified`/`task_failed` events
carry the 1-based `attempt` and the number of `retries_left` (0 when exhausted)
so a consumer can reconstruct exactly how many bounded retries ran.

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
- **Git dirty-state guard** (BUILD/AUTO): files with pre-existing uncommitted
  changes at run start are protected — ASCS will not overwrite existing user
  work without explicit approval.

## Diagnostics (`risa doctor`)

```
risa --doctor [--workspace <path>]
```

Runs a read-only diagnostic suite and reports a `PASS` / `WARN` / `FAIL`
status for each item, with actionable recovery hints:

- Python environment and A.S.C.S. install
- Configuration validity
- Workspace existence and containment
- Ollama connectivity, the configured model **installed**, and a live
  **model-query probe** (a slow/unresponsive local model is reported as a WARN)
- Registered tools
- Persistent context index health
- Project manifest / discovery state
- Git availability
- pytest availability
- **Task engine** integrity (planner + executor importable; task-graph model
  round-trips through persistence)

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

## Experience memory

Beyond per-project state, ASCS keeps a cross-project **learning memory**
(`agent/experience/store.py`). Every completed run records a structured experience —
task, outcome, the ordered plan, actions taken, model-visible observations,
verification result, and a `success`/`score` — to a local JSONL store
(`~/.risa/ascs/experiences.jsonl` by default).

- **Reuse wins** — on later runs, a short, bounded "PAST EXPERIENCE" block of
  high-scoring, overlapping prior outcomes is injected into the system prompt
  and the planner prompt, so the model starts from approaches that already
  worked instead of rediscovering them.
- **Avoid repeated failures** — when a run fails, experiences that previously
  claimed success on the same task are *penalised* (score lowered, floor −1.0),
  so the model stops trusting an approach that just contradicted it.
- **Controls** — the CLI enables this by default (`AGENT_EXPERIENCE_ENABLED=true`
  when launched through `risa`; library callers opt in on `AgentConfig`).
  Point it elsewhere with `AGENT_EXPERIENCE_PATH`, or disable with
  `AGENT_EXPERIENCE_ENABLED=false` / `AGENT_EXPERIENCE_ENABLED=0`.
- **Bounded injection** — only the top few matching experiences (a few KB) are
  ever sent; the full history stays on disk and is never loaded into context.

## Task engine

Plans and work items are structured as a dependency-aware **task graph**
(`agent.tasks`) with explicit statuses
(`pending → ready → running → completed / failed / blocked / cancelled /
skipped`). Each task carries its own description, dependencies, files,
commands, verification list, retry count and failure reason, plus a
`complexity` (small/medium/large) and `kind` (inspect/plan/implement/verify/
review).

The engine is fully implemented and reachable via `risa --tasks "…"`:

1. **Plan** (`agent.planner`) — the model decomposes the objective into a
   validated DAG (fan-in/fan-out supported), seeded by the project manifest and
   hierarchical retrieval. Oversized (`large`) tasks are automatically
   re-chunked into per-file subtasks, and every task is guaranteed a
   verification step.
2. **Execute** (`agent.executor`) — `agent/execution/executor.py` walks the graph, runs
   each ready task with a **task-scoped system prompt**, then verifies it
   against its acceptance criteria (each `run …` verification step must exit 0)
   before marking it complete. Implementing tasks with no declared verification
   steps are treated as **not fully verified** (not silently passing). A
   verification failure is fed back to the model for a bounded retry
   (`max_verify_retries=2`, configurable via `AGENT_MAX_VERIFY_RETRIES`);
   retries exhausted → task `FAILED`, dependents cascade-cancelled.
3. **Mode gating** — the task engine respects PLAN/BUILD/SAFE modes:
   - **PLAN**: read-only tools only (no file writes, no `run_command`, no
     verification commands). The planner still produces and renders the task
     graph/plan.
   - **BUILD/AUTO**: full modification allowed. **Git dirty-state guard**:
     files dirty at run start are protected; writes to pre-existing dirty
     files are blocked with a clear "protected" message.
   - **SAFE**: every modifying tool and verification command requires operator
     approval before execution.
4. **Per-task action log** — each task records a structured log of every
   tool/command it ran, files affected, and verification result, surfaced
   through the step log and events for human and machine readability.
5. **Inspect & resume** — progress is persisted to `.ascs/task_state.json`
   after every task; a run can be resumed from that state. A task stuck in
   `RUNNING` by an interrupted run is automatically reset to `READY` so resume
   can retry it without restarting completed work. The plan and each task are
   surfaced through `task_plan` / `task_started` / `task_verified` /
   `task_failed` / `task_completed` events plus a human-readable step log.

The classic single-shot loop (`risa "…"`, no `--tasks`) remains the default
mode and is unchanged.

## Configuration

Environment variables (CLI flags win, both override defaults):

| Variable                 | Meaning                              | Default               |
| ------------------------ | ------------------------------------ | --------------------- |
| `AGENT_MODE`             | `PLAN` / `BUILD` / `AUTO` / `SAFE`   | `AUTO`                |
| `AGENT_APPROVAL`         | in SAFE mode, auto-approve all       | off (ask each time)   |
| `AGENT_UI_HOST` / `AGENT_UI_PORT` | web UI bind address / port | `127.0.0.1` / `8787` |
| `OLLAMA_BASE_URL`        | Ollama server URL                    | `http://localhost:11434` |
| `OLLAMA_MODEL`           | Ollama model                         | `qwen3-coder:30b` (fallback `qwen2.5-coder:14b`) |
| `AGENT_MAX_ITERATIONS`   | agent iteration budget (→ `TIMEOUT`) | `50`                  |
| `AGENT_REQUEST_TIMEOUT`  | per-model-call timeout (s)           | `600`                 |
| `AGENT_COMMAND_TIMEOUT`  | default `run_command` timeout (s)    | `120`                 |
| `AGENT_KEEP_ALIVE`       | Ollama keep-alive (e.g. `30m`)       | `30m`                 |
| `AGENT_PREWARM`          | warm the model at startup            | on                    |
| `AGENT_VERBOSE`          | show tool output in CLI logs         | off                   |
| `AGENT_MAX_OUTPUT_CHARS` | per-tool-output truncation limit     | `20000`               |
| `AGENT_CONTEXT_BUDGET_CHARS` | rolling conversation budget      | `70000`               |
| `AGENT_MALFORMED_RETRY_LIMIT` | bad-reply retries before `FAILED` | `5`               |
| `AGENT_MAX_VERIFY_RETRIES` | verification failure retries per task | `2`           |
| `AGENT_NUM_CTX`          | Qwen3 context window size (max-chunking: 65k default, 131k xhigh for 300k shards) | `65536`               |
| `AGENT_NUM_PREDICT`      | max tokens generated per request   | `16384`               |
| `AGENT_MAX_RETRIES`      | retries past the initial attempt   | `2`                   |
| `AGENT_BACKOFF_S`        | base backoff between retries (s)   | `2.0`                 |
| `AGENT_EXPERIENCE_ENABLED` | write + reuse experience memory    | `true` (CLI)          |
| `AGENT_EXPERIENCE_PATH`  | JSONL store for verified outcomes    | `~/.risa/ascs/`       |

The `AGENT_NUM_CTX` / `AGENT_NUM_PREDICT` values are sent **only** to the native
Ollama `/api/chat` endpoint (as `options`); they are never forwarded to an
OpenAI-compatible endpoint. See `docs/OLLAMA_SETUP.md` for the full local-model
setup and verification, including the OpenCode `ollama/qwen3-coder:30b` + fallback `qwen2.5-coder:14b` config. Max-chunking splits a 300k-token objective into feasible `~8k` shards within the 65k window.

## Development

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt   # or: python -m pip install -e ".[dev]"
python -m pytest                            # venv python on Windows; macOS/Linux devs: .venv/bin/python -m pytest -q (python -> python3 fallback automatic)
```

> Cross-platform dev: `pytest` is cross-platform (macOS/Linux fallback `python` -> `python3` is automatic in `agent/tools/core.py:508`). The full runtime (`risa --ui`/`--tui`) remains Windows-only.

All tests run offline against scripted fake clients and an in-process mock
Ollama server — **no Ollama required**. The skipped tests are the opt-in live
checks that need a real local `qwen3-coder:30b` (or `qwen2.5-coder:14b` fallback) and a running Ollama server,
gated behind `RISALIVE=1`:

```powershell
$env:RISALIVE="1"; pytest -q tests/test_ollama_live.py   # bash: RISALIVE=1 pytest -q ...
$env:RISALIVE=""
```

On a machine with a running Ollama and `qwen3-coder:30b` installed these take
minutes; otherwise they skip (use `--model qwen2.5-coder:14b` on 16GB). See `docs/OLLAMA_SETUP.md` for the full Windows
setup and verification (32 GB recommended for 30B; 300k tasks rely on max-chunking).

Test suite covers the tool layer, the loop's lifecycle/plan/mode/cancellation
behaviour (using scripted fake clients — no Ollama required), event emission,
the Ollama HTTP client against an in-process mock server, the state machine,
the staged boot, the web server endpoints, and the task engine (planner,
executor, task-graph DAG behaviour, mode gating, git-dirty protection,
verification retry, interrupt+resume, and end-to-end pipeline integration).

## Scratch outputs

Model-generated scratch/demo/test outputs go in `Ollama_tests/` (gitignored
sandbox, never committed). Keep real source edits in their normal repo paths.