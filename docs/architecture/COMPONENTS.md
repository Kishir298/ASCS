# A.S.C.S. Components

Canonical module → responsibility, with the old import path where a shim
preserves it. Line counts from the Phase 0 post-migration tree; behavior
unchanged in Phase 0.

## Core (`agent.core`)

- `agent/core/loop.py` (1007 lines) — `AgentLoop(run/run_graph)`,
  `run_agent/run_graph_agent`, SAFE approval, malformed/repeat guards,
  context trim, cancellation. The brain; Phase 1 audit target (do not fix
  here). Old path: `agent/loop.py` (shim, 13 lines).
- `agent/core/state.py` (230 lines) — `StateTracker`, lifecycle constants,
  `StateSnapshot`. Old path: `agent/state.py` (shim, 46 lines).
- `agent/events.py` (534 lines) — shared observability via `agent.core`:
  `AgentEvent`, `EventSink`, `emit_*` helpers.

## Planning (`agent.planning`)

- `agent/planning/planner.py` (390 lines) — `plan_objective`,
  `planner_prompt`, `project_intelligence`, `parse_tasks`,
  `_derive_verification`, `plan_text`. Large-task auto-chunking; every task
  gets verification. Old path: `agent/planner.py` (shim).
- `agent/planning/prompts.py` (207 lines) — `system_prompt`, `task_message`,
  `malformed_feedback`, `tool_error_feedback`; per-mode instructions; Windows
  `cmd.exe` facts; `PROJECT INTELLIGENCE` + `PAST EXPERIENCE` injection.
  Old path: `agent/prompts.py` (shim).
- `agent/models/responses.py` (242 lines) — `ModelReply`, `Plan`,
  `ToolResult`, `parse_model_reply` (fence/prose tolerant), `truncate`.

## Execution (`agent.execution`)

- `agent/execution/tasks.py` (464 lines) — `Task`, `TaskGraph`
  (topo/ready/recompute/cascade), statuses, `build_graph_from_specs`,
  `chunk_graph`, `plan_to_graph`; persisted to `.ascs/task_state.json`.
  Old path: `agent/tasks.py` (shim).
- `agent/execution/executor.py` (819 lines) — `TaskExecutor`, task-scoped
  prompts, `_verify_task` (`run …` must exit 0), `max_verify_retries=2`,
  PLAN/SAFE and git-dirty gating, persistence + re-index. Old path:
  `agent/executor.py` (shim).

## Tools (`agent.tools`)

- `agent/tools/core.py` (955 lines) — `TOOL_SPECS` (13 tools),
  `validate_tool_call`, `execute_tool → ToolResult`, `_execute_process`
  (PATH injection, `python → python3` POSIX fallback at line ~508,
  `taskkill /T`), truncation, traversal and VCS guards, `set_plan`,
  `inspect_environment`, `git_status`/`git_diff`. Old import path
  `agent.tools` preserved via lazy re-exports in the package `__init__`.

## Context (`agent.context`)

- `agent/context/index.py` (1265 lines) — `ProjectIndex` (build/update/
  search, L1–L4 `retrieve`, `chunks_for_file`),
  `FileRecord/Symbol/ContextChunk/Bundle`, `chunk_tokens_for`
  (30b 8192 / 14b 4096), `.ascs/context_index.json` incremental. Old path:
  `agent/context.py` (removed; old API re-exported by the package `__init__`).
- `agent/context/project.py` (610 lines) — `ProjectScanner/scan`,
  `ProjectStore` (`.ascs/project_manifest.json`, task-graph persist),
  `project_prompt_text`. Old path: `agent/project.py` (shim).
- `agent/context/toolchain.py` (210 lines) — `detect_toolchain`
  (`pyproject > requirements > package.json > Cargo > go.mod`),
  `toolchain_to_text`. Old path: `agent/toolchain.py` (shim).

## Experience (`agent.experience`)

- `agent/experience/store.py` (523 lines) — `Experience`, `ExperienceStore`
  (`~/.risa/ascs/experiences.jsonl`, max 5000, corrupt-skip),
  `save_run/search/penalize_contradictions/update_feedback`,
  `format_for_prompt` (bounded KB injection). Old path:
  `agent/experience.py` (removed; re-exported by the package `__init__`).

## Verification (`agent.verification`)

- Distributed in Phase 0: executor verify/retry/cascade +
  toolchain-derived acceptance + loop guards + `verification_started` /
  `retry` / `task_verified` / `task_failed` events. The package re-exports
  `TaskOutcome` / `VerificationResult` / `TaskActionLog` and marks the home
  for Phase 6 hardening; no logic moved in Phase 0.

## Models (`agent.models`)

- `agent/models/client.py` (823 lines) — `OllamaClient`
  (`chat/chat_resilient/chat_stream/ensure_ready/list_models/warm/
  abort_current`), error taxonomy, `<think>` stripping, NDJSON tolerance,
  retry 2× + backoff. Old path: `agent/ollama.py` (shim).
- `agent/models/providers.py` (412 lines) — `PROVIDER_NAMES`,
  `list_models_for_provider`, cached parallel listing, Ollama-compat
  fallback; backs TUI `/models` `/connect`. Old path: `agent/providers.py`
  (shim).
- `agent/models/responses.py` (242 lines) — response contract as above
  (planning section). Old path: `agent/models.py` (removed; re-exported by
  the package `__init__`).

## Terminal (`agent.terminal`)

- `agent/terminal/entry.py` (112 lines) — `normalize_argv`, `ASCII_BANNER`,
  `main`. Old path: `agent/terminal.py` (removed; re-exported lazily by the
  package `__init__` and preserved as `python -m agent.terminal` via
  `__main__.py`).
- `agent/terminal/tui.py` (3978 lines) — `TuiApp`, `run_tui`, layout tiers,
  geometry, themes, TAB `PLAN → BUILD → AUTO`, slash commands, pickers,
  `TaskRunner` wiring, persisted state. Old path: `agent/tui.py` (shim).
- `agent/web.py` (528 lines) — `App/TaskRunner/EventHub`, SSE routes; HTTP
  serving legacy (missing `agent/ui/index.html` → `_fallback_html()`), but
  `EventHub`/`TaskRunner` shared with the TUI — do not delete.

## Support (top-level)

- `agent/config.py` (499 lines) — frozen `AgentConfig`, `load_config`
  (CLI > env > TUI-state > defaults), `MODES`, `MODIFY/READONLY_TOOLS`,
  `INTELLIGENCE_MAP`, defaults (`qwen3-coder:30b`, `65536/16384`, `600s`,
  `30m` keep-alive).
- `agent/workspace.py` (193 lines) — `Workspace(root)`, `resolve`
  containment, `should_ignore`, `iter_files`.
- `agent/boot.py` (268 lines), `agent/doctor.py` (380 lines),
  `agent/main.py` (458 lines) as in `RUNTIME.md`.
