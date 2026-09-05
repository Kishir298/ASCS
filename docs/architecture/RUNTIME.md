# A.S.C.S. Runtime

## Entrypoints (actual)

- `risa` → `agent.terminal:main` (`pyproject.toml` `[project.scripts]`). The
  terminal package re-exports `main` lazily from
  `agent/terminal/entry.py::main` (moved from `agent/terminal.py` in Phase 0).
- `python -m agent.terminal` → `agent/terminal/__main__.py` →
  `agent.terminal.entry:main`.
- `python -m agent` → `agent/__main__.py` → `agent.main:main`.
- `agent/terminal/entry.py::normalize_argv` appends `--tui` unless a
  diagnostic (`--check`, `--doctor`, `--list-models`) was given; `--ui` is
  dropped and treated as `--tui`.
- `agent/main.py::build_parser` defines `task`, `--mode plan|build|auto`,
  `--safe`, `--tasks`, `--tui`, `--ui`, `--check`, `--doctor`,
  `--list-models`, `--model`, `--base-url`, `--intelligence`, `--provider`,
  plus generation knobs. Dispatch: `_cmd_check` / `_cmd_doctor` /
  `_cmd_list_models` / `_cmd_tui` / `run_agent` / `run_graph_agent`.
- `risa.cmd` is an AppControl-safe shim (`.venv\Scripts\python.exe -m agent`).

## Startup

`agent/boot.py::boot` runs staged checks with real timing and recovery hints:
`config → Python 3.12+ → workspace → Ollama connectivity → model availability
→ warm → tools → writable env + pytest`. `risa --doctor` (`agent/doctor.py`)
is the read-only counterpart (13 checks, model-query probe is WARN not FAIL).

## Loop and state

- `agent/core/loop.py::AgentLoop` (single-shot `run`, graph `run_graph`)
  drives `task → system+project+experience prompt → chat_resilient → parse →
  execute_tool → verify` until done or limits (`AGENT_MAX_ITERATIONS=50`,
  malformed-retry 5, context budget 70k chars, per-call timeout 600s).
- `agent/core/state.py::StateTracker` enforces
  `RECEIVING_TASK → PLANNING → EXECUTING → VERIFYING → COMPLETE`
  (or `FAILED` / `CANCELLED` / `TIMEOUT`), thread-safe, shared by loop/TUI.
- `agent/events.py` emits JSON-serializable operational events only (no
  chain-of-thought): lifecycle, model, tool, command, test, file, task-graph,
  retry/verification. Consumed by TUI/EventHub and the (legacy) web SSE feed.

## Domain packages (Phase 0 boundaries)

Phase 0 moved canonical implementations into domain subpackages. Old flat
import paths (`agent.loop`, `agent.planner`, `agent.ollama`, …) keep working
through thin shims that re-export the canonical module — no duplicated logic,
no behavior change. New code should import from the domain packages.

- `agent.core` — canonical: `loop.py` (`AgentLoop`, `LoopResult`,
  `run_agent`, `run_graph_agent`), `state.py` (`StateTracker`,
  `StateSnapshot`, lifecycle constants). `events.py` stays top-level as
  shared observability and is re-exported here. Shims: `agent/loop.py`,
  `agent/state.py`.
- `agent.planning` — canonical: `planner.py` (objective → validated DAG,
  large-task chunking, verification guarantees), `prompts.py` (system/task
  prompts, per-mode instructions, `PROJECT INTELLIGENCE` + `PAST EXPERIENCE`
  injection). Shims: `agent/planner.py`, `agent/prompts.py`. Planning-side
  response parsing lives in `agent/models/responses.py`.
- `agent.execution` — canonical: `tasks.py` (`Task`, `TaskGraph`,
  statuses, chunking, persisted to `.ascs/task_state.json`), `executor.py`
  (`TaskExecutor`, task-scoped prompts, `_verify_task`, bounded retries,
  cascade-cancel, PLAN/SAFE + git-dirty gating). Shims: `agent/tasks.py`,
  `agent/executor.py`.
- `agent.tools` — canonical: `core.py` (13 tools: filesystem, search, git,
  shell, environment, `set_plan`; containment, truncation, timeouts,
  `python → python3` POSIX dev fallback, dirty-guard). Old import path
  preserved by lazy re-exports in the package `__init__` (no shim file:
  the old `agent.tools` module *is* this package now).
- `agent.context` — canonical: `index.py` (`ProjectIndex`, L1–L4 retrieval,
  `chunks_for_file`, model-specific chunk tokens 30b 8192 / 14b 4096,
  `.ascs/context_index.json`), `project.py` (`ProjectScanner`,
  `ProjectStore`, `.ascs/project_manifest.json`), `toolchain.py`
  (evidence-based command derivation). Old `agent.project` /
  `agent.toolchain` paths have shims; the old `agent.context` index API is
  re-exported lazily by the package `__init__`.
- `agent.experience` — canonical: `store.py` (JSONL store at
  `~/.risa/ascs/experiences.jsonl`, search, contradiction penalties, bounded
  prompt formatting). Old `from agent.experience import …` works via lazy
  re-exports. Phase 5 owns the learning architecture; Phase 0 only provides
  the home.
- `agent.verification` — boundary for the distributed verify → retry →
  fail/cascade flow (executor + toolchain + loop guards + events); re-exports
  `TaskOutcome` / `VerificationResult` / `TaskActionLog`. No new logic in
  Phase 0.
- `agent.models` — canonical: `client.py` (stdlib urllib `OllamaClient`;
  `num_ctx`/`num_predict` sent only on native `/api/chat`; retry + backoff;
  `<think>` stripping), `providers.py` (multi-provider listing, Ollama
  fallback; backs TUI `/models` `/connect`), `responses.py` (strict
  `{comment,tool,arguments} | {done:true,summary}` contract). Shims:
  `agent/ollama.py`, `agent/providers.py`; the old `agent.models` response
  API is re-exported lazily by the package `__init__`.
- `agent.terminal` — canonical: `entry.py` (argv normalization, banner,
  `main`), `tui.py` (curses UI, slash commands, pickers, geometry tiers),
  `__main__.py`. Shared `EventHub` / `TaskRunner` live in `agent/web.py`
  (HTTP serving itself is legacy with a missing-asset fallback). Shim:
  `agent/tui.py`; the old `agent.terminal` entry API is re-exported lazily.

Top-level survivors (real runtime code): `config.py`, `main.py`,
`workspace.py`, `boot.py`, `doctor.py`, `events.py`, `web.py`. Every other
`.py` directly under `agent/` is a compatibility shim (13–46 lines).
