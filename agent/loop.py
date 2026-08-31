"""The autonomous agent loop.

Drives: user task -> model analysis -> tool selection -> execution -> result
-> model analysis -> ... until completion, iteration limit, interruption, or a
fatal error.

Lifecycle
=========
Every run moves through an explicit state machine (``agent.state``):

    RECEIVING_TASK -> PLANNING -> EXECUTING -> VERIFYING -> COMPLETE

with FAILED / CANCELLED / TIMEOUT as the terminal abort states. State changes
are pushed to the :class:`StateTracker` and broadcast as ``status`` events.

Modes
=====
    PLAN  - only inspection + set_plan tools are enabled; no writes.
    BUILD - plan then implement; modification/command tools may require
            operator approval when config.approval is set (legacy SAFE).
    AUTO  - fully autonomous end-to-end.

Cancellation
============
The loop checks ``should_stop()`` before every model call and after every tool
call. A ``KeyboardInterrupt`` raised into the running thread (or Ctrl+C in a
terminal) is converted to a clean ``cancelled`` result that kills any active
subprocess tree and releases the worker.
"""

from __future__ import annotations

import json as _json
import re
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import state as _state
from .config import AgentConfig, MODIFY_TOOLS
from .executor import TaskExecutor, TaskExecution
from .events import (
    EventSink,
    emit_activity,
    emit_command_completed,
    emit_command_output,
    emit_command_started,
    emit_completed,
    emit_error,
    emit_file_event,
    emit_mode_changed,
    emit_model_completed,
    emit_model_started,
    emit_started,
    emit_status,
    emit_stopped,
    emit_task_completed,
    emit_task_plan,
    emit_task_started,
    emit_test_completed,
    emit_test_started,
    emit_tool_completed,
    emit_tool_started,
    null_sink,
)
from .models import Plan, ToolResult, parse_model_reply, tool_result_message
from .ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from .planner import plan_objective, plan_text
from .prompts import (
    malformed_feedback,
    system_prompt,
    task_message,
    tool_error_feedback,
)
from .state import StateTracker
from .tasks import Task
from .tools import execute_tool
from .workspace import Workspace

DEFAULT_APPROVER: Callable[[str], bool] = lambda desc: (
    input(f"[SAFE] Approve: {desc} [y/N] ").strip().lower() in ("y", "yes")
)

FILE_EVENT_FOR_TOOL = {
    "read_file": "file_read",
    "write_file": "file_written",
    "apply_patch": "patch_applied",
    "delete_file": "file_deleted",
    "move_file": "file_moved",
    "copy_file": "file_copied",
}

# Heuristic for VERIFYING state: commands that clearly run the test suite.
_TEST_COMMAND_RE = re.compile(
    r"(^|\s)("
    r"pytest|unittest|py\.test|tox|nox|mvn\s+test|go\s+test|npm\s+test|"
    r"python\s+(-m\s+)?pytest|py\s+(-m\s+)?pytest"
    r")(\s|$|-)",
    re.IGNORECASE,
)

RETRY_PROMPT = (
    "The model returned an unusable response. Reply with ONLY a valid JSON "
    "tool call or done object as instructed."
)


def _content(message: dict[str, str]) -> str:
    """Read the ``content`` field robustly (guards against malformed data)."""
    if not isinstance(message, dict):
        return ""
    value = message.get("content")
    return value if isinstance(value, str) else ""


def is_test_command(command: str) -> bool:
    """True when ``command`` looks like it runs the test suite."""
    return bool(_TEST_COMMAND_RE.search(command or ""))


def _chat_value(text: str):
    """Extract a JSON value (object or array) from a planner reply, else raw text."""
    import json

    raw = (text or "").strip()
    if not raw:
        return None
    for opener in ("{", "["):
        start = raw.find(opener)
        if start < 0:
            continue
        try:
            return json.JSONDecoder().raw_decode(raw[start:].lstrip())[0]
        except (json.JSONDecodeError, ValueError):
            continue
    return raw


def _resilient_chat(client, messages: list[dict[str, str]], **kwargs) -> str:
    """Issue a chat request, using the resilient (retrying) path when available.

    Real :class:`~agent.ollama.OllamaClient` instances provide
    ``chat_resilient`` (bounded retry on transient connection/5xx errors);
    test fakes fall back to their plain ``chat`` (which does not accept the
    ``should_stop`` retry hook, so that kwarg is stripped).
    """
    if hasattr(client, "chat_resilient"):
        return client.chat_resilient(messages, **kwargs)
    chat = getattr(client, "chat")
    kwargs.pop("should_stop", None)
    return chat(messages, **kwargs)


@dataclass
class LoopResult:
    status: str = "interrupted"  # completed|max_iterations|interrupted|cancelled|fatal|malformed
    state: str = _state.CANCELLED  # lifecycle terminal state
    summary: str = ""
    iterations: int = 0
    steps: list[str] = field(default_factory=list)
    error: str = ""
    plan: Plan | None = None

    @property
    def is_complete(self) -> bool:
        return self.status == "completed"


@dataclass
class GraphLoopResult:
    """Outcome of a task-graph-driven run (planner + executor)."""

    status: str = "completed"  # completed|partial|cancelled|failed|fatal
    objective: str = ""
    summary: str = ""
    iterations: int = 0
    plan: str = ""  # human-readable plan (plan_text)
    steps: list[str] = field(default_factory=list)
    task_count: int = 0
    progress: dict = field(default_factory=dict)
    error: str = ""
    report: str = ""  # truthful end-of-run report

    @property
    def is_complete(self) -> bool:
        return self.status == "completed"


class AgentLoop:
    """Runs one agent session against a workspace."""

    def __init__(
        self,
        config: AgentConfig,
        client: OllamaClient,
        workspace: Workspace,
        *,
        approver: Callable[[str], bool] | None = None,
        log: Callable[[str], None] | None = None,
        event_sink: EventSink | None = None,
        should_stop: Callable[[], bool] | None = None,
        tracker: StateTracker | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.ws = workspace
        self.log = log or (lambda _msg: None)
        self.event_sink = event_sink or null_sink
        self.approver = approver if approver is not None else DEFAULT_APPROVER
        self.should_stop = should_stop or (lambda: False)
        self.tracker = tracker or StateTracker(_state.IDLE)
        self._steps: list[str] = []
        self._messages: list[dict[str, str]] = []
        self._malformed_count = 0
        self._last_call: tuple[Any, ...] | None = None
        self._last_ok = True
        self._repeat_count = 0
        self._plan: Plan | None = None

    # -- logging / events ---------------------------------------------------

    def _step(self, text: str) -> None:
        self._steps.append(text)
        self.log(text)

    def _set_state(self, state: str, message: str = "") -> None:
        """Transition lifecycle state + broadcast a status event."""
        self.tracker.set(state, message)
        emit_status(
            self.event_sink,
            _state.STATE_LABELS.get(state, state.upper()),
            message or _state.STATE_LABELS.get(state, state.upper()),
        )

    def _emit_command(self, event_name: str, command: str, *, exit_code: int | None = None, elapsed: float | None = None) -> None:
        if event_name == "start":
            emit_command_started(self.event_sink, command)
            if is_test_command(command):
                emit_test_started(self.event_sink, command)
        else:
            rc = exit_code if exit_code is not None else 0
            ok = rc == 0
            emit_command_completed(
                self.event_sink, command, exit_code=rc, ok=ok, elapsed=elapsed
            )
            if is_test_command(command):
                emit_test_completed(
                    self.event_sink, command, exit_code=rc, ok=ok, elapsed=elapsed
                )

    # -- main entry ---------------------------------------------------------

    def _project_prompt_block(self) -> str:
        """Build the project-intelligence block for the system prompt.

        Best-effort: if project discovery fails, the agent still runs with a
        generic prompt rather than being blocked by an indexing error.
        """
        try:
            from .project import ProjectStore, project_prompt_text

            store = ProjectStore(self.ws.root)
            store.refresh()
            return project_prompt_text(store)
        except Exception as exc:  # noqa: BLE001 - discovery must never block the agent
            return f"- Project discovery unavailable: {exc}"

    def run(self, task: str) -> LoopResult:
        self._messages = [
            {
                "role": "system",
                "content": system_prompt(self.config, self._project_prompt_block()),
            },
            task_message(task),
        ]
        self.tracker.configure(mode=self.config.mode, task=task)
        self._set_state(_state.RECEIVING_TASK, "Task received")
        emit_started(self.event_sink, "A.S.C.S. agent started")
        emit_mode_changed(self.event_sink, self.config.mode)

        self._step(f"Task received: {task[:300]}")
        self._step(f"Workspace: {self.ws.root}")
        self._step(
            f"Model: {self.client.model}  Mode: {self.config.primary_mode}  "
            f"Tools: {len(self.config.effective_tools)} available  "
            f"Max iterations: {self.config.max_iterations}"
        )
        self.tracker.start(_state.RECEIVING_TASK)

        iteration = 0
        try:
            while iteration < self.config.max_iterations:
                if self.should_stop():
                    return self._finish(
                        "cancelled",
                        "Stopped by the operator.",
                        iteration,
                        _state.CANCELLED,
                    )
                iteration += 1
                self._step(f"[{iteration:02d}] Asking model for the next step...")
                if self.tracker.state in (_state.RECEIVING_TASK, _state.VERIFYING):
                    self._set_state(_state.PLANNING, "Model is analysing the result")

                thinking_start = _time.monotonic()
                try:
                    emit_model_started(
                        self.event_sink, "Model is thinking/generating..."
                    )
                    reply_text = _resilient_chat(
                        self.client,
                        self._messages_for_request(),
                        format="json",
                        should_stop=self.should_stop,
                    )
                    emit_model_completed(
                        self.event_sink,
                        "Model responded",
                        elapsed=_time.monotonic() - thinking_start,
                    )
                except OllamaResponseError as exc:
                    self._step(f"[{iteration:02d}] Unusable model response: {exc}")
                    self._messages.append(
                        {"role": "assistant", "content": f"(empty/invalid response: {exc})"}
                    )
                    self._messages.append({"role": "user", "content": RETRY_PROMPT})
                    if self._bump_malformed(iteration):
                        return self._finish(
                            "malformed",
                            "Model repeatedly produced unusable responses.",
                            iteration,
                            _state.FAILED,
                            error=str(exc),
                        )
                    continue

                if self.should_stop():
                    return self._finish(
                        "cancelled",
                        "Stopped by the operator mid-analysis.",
                        iteration,
                        _state.CANCELLED,
                    )

                reply = parse_model_reply(reply_text)

                if reply.error is not None:
                    self._step(f"[{iteration:02d}] Model reply rejected: {reply.error}")
                    self._messages.append({"role": "assistant", "content": reply_text})
                    self._messages.append(malformed_feedback(reply.error))
                    if self._bump_malformed(iteration):
                        return self._finish(
                            "malformed",
                            "Model repeatedly failed to produce a valid response.",
                            iteration,
                            _state.FAILED,
                            error=reply.error,
                        )
                    continue

                if reply.comment:
                    self._step(f"[{iteration:02d}] {reply.comment[:300]}")
                    emit_activity(self.event_sink, reply.comment[:300])

                if reply.done:
                    self._step("[done] Model reports the task is complete.")
                    return self._finish(
                        "completed",
                        reply.summary,
                        iteration,
                        _state.COMPLETE,
                        plan=self._plan,
                    )

                enabled = list(self.config.effective_tools)
                if reply.tool not in enabled:
                    valid = ", ".join(enabled)
                    self._step(
                        f"[{iteration:02d}] Tool '{reply.tool}' is not enabled; "
                        f"enabled tools: {valid}"
                    )
                    self._messages.append({"role": "assistant", "content": reply_text})
                    self._messages.append(
                        tool_error_feedback(
                            reply.tool or "",
                            f"tool not available in this session/mode (available: {valid}).",
                        )
                    )
                    continue

                if self._is_repeated_call(reply.tool, reply.arguments):
                    # Only identical, CONSECUTIVE, FAILING calls are treated as
                    # a stuck loop. Identical successful calls are harmless.
                    if self._last_ok:
                        self._repeat_count = 0
                    else:
                        self._repeat_count += 1
                        if self._repeat_count >= 2:
                            self._step(
                                f"[{iteration:02d}] Identical failing call repeated too many times."
                            )
                            return self._finish(
                                "fatal",
                                "The model repeated an identical failing tool call; "
                                "stopping to avoid an infinite loop.",
                                iteration,
                                _state.FAILED,
                            )
                else:
                    self._repeat_count = 0

                self._last_call = (reply.tool, tuple(sorted(reply.arguments.items())))
                display_args = _json.dumps(reply.arguments, sort_keys=True)

                result = self._run_tool(reply.tool, reply.arguments, iteration)

                if result.ok:
                    self._malformed_count = 0
                self._last_ok = result.ok

                self._messages.append({"role": "assistant", "content": reply_text})
                self._messages.append(tool_result_message(result))

            return self._finish(
                "max_iterations",
                f"Stopped after {iteration} iterations "
                f"(AGENT_MAX_ITERATIONS={self.config.max_iterations}).",
                iteration,
                _state.TIMEOUT,
                plan=self._plan,
            )
        except KeyboardInterrupt:
            if self.should_stop():
                self._step("[stop] Stopped by the operator.")
                return self._finish(
                    "cancelled",
                    "Cancelled by the operator.",
                    iteration,
                    _state.CANCELLED,
                )
            self._step("[interrupt] Stopped by user (Ctrl+C).")
            return self._finish(
                "interrupted",
                "Interrupted by the user.",
                iteration + 1,
                _state.CANCELLED,
            )
        except OllamaTimeoutError as exc:
            return self._finish(
                "fatal", "Ollama request timed out.", iteration, _state.FAILED, error=str(exc)
            )
        except OllamaConnectionError as exc:
            return self._finish(
                "fatal", "Ollama is unavailable.", iteration, _state.FAILED, error=str(exc)
            )
        except OllamaModelNotFoundError as exc:
            return self._finish(
                "fatal",
                f"Model '{self.client.model}' is not installed on the Ollama server.",
                iteration,
                _state.FAILED,
                error=str(exc),
            )
        except OllamaError as exc:
            return self._finish(
                "fatal", "Ollama request failed.", iteration, _state.FAILED, error=str(exc)
            )

    # -- task-graph driven execution (planner + executor) -------------------

    def run_graph(
        self,
        objective: str,
        *,
        resume: bool = False,
        graph: Task | None = None,
    ) -> GraphLoopResult:
        """Plan ``objective`` into a task graph and execute it task-by-task.

        This is the complementary task-driven path to :meth:`run`. It:

        * plans the objective into a validated, size-controlled
          :class:`~agent.tasks.TaskGraph` via :func:`~agent.planner.plan_objective`,
        * persists progress to ``.ascs/task_state.json`` and can resume an
          in-progress run,
        * emits task-plan / task-started / task-completed inspection events and
          keeps a human-readable step log,
        * verifies each task against its acceptance criteria before marking it
          complete.

        ``graph`` is a reserved hook (currently unused) so callers may later
        inject a prepared graph; the normal flow plans it from ``objective``.
        """
        self.tracker.configure(mode=self.config.mode, task=objective)
        emit_started(self.event_sink, "A.S.C.S. task engine started")
        emit_mode_changed(self.event_sink, self.config.mode)
        self._step(f"Objective: {objective[:300]}")
        self._step(f"Workspace: {self.ws.root}")
        self._step(
            f"Model: {self.client.model}  Mode: {self.config.primary_mode}  "
            f"Task engine: on  Max iterations/task: {self.config.max_iterations}"
        )
        self.tracker.start(_state.PLANNING)

        from .project import ProjectStore

        store = ProjectStore(self.ws.root)
        store.refresh()

        result = GraphLoopResult(objective=objective)

        try:
            planned = self._plan_objective(objective, store)
            result.plan = planned["text"]
            result.task_count = planned["count"]
            self._step("Plan:")
            for line in result.plan.splitlines():
                self._step(f"  {line}")
            emit_task_plan(self.event_sink, result.plan, task_count=result.task_count)

            target = planned["graph"]
            if resume:
                persisted = store.load_task_graph()
                if persisted and len(persisted) and not persisted.all_complete:
                    target = persisted
                    from .executor import TaskExecutor as _TE

                    reset = _TE.reset_stuck_tasks(target)
                    if reset:
                        self._step(
                            f"Reset stuck RUNNING task(s) for resume: "
                            f"{', '.join(reset)}"
                        )
                    self._step("Resuming an in-progress task graph from .ascs/task_state.json")

            executor = self._build_executor(store)
            if executor.git_baseline:
                self._step(
                    f"Git dirty baseline ({len(executor.git_baseline)} file(s)): "
                    f"{', '.join(sorted(executor.git_baseline)[:10])}"
                )
            execution: TaskExecution = executor.execute(objective, target)

            result.status = execution.status
            result.summary = execution.summary
            result.iterations = execution.iterations
            result.progress = execution.progress
            result.report = execution.report()
            result.steps = list(self._steps)
            if execution.status in ("completed", "partial") and execution.failed_tasks:
                result.status = "partial"

            if execution.status == "completed" and result.summary:
                self._set_state(_state.COMPLETE, result.summary)
            elif execution.status == "cancelled":
                self._set_state(_state.CANCELLED, result.summary)
            else:
                self._set_state(_state.FAILED, result.summary)

            return result
        except OllamaModelNotFoundError as exc:
            result.status = "fatal"
            result.error = f"Model '{self.client.model}' is not installed on the Ollama server."
            self._step(f"[error] {result.error}")
            self._set_state(_state.FAILED, result.error)
            return result
        except OllamaError as exc:
            result.status = "fatal"
            result.error = f"Ollama request failed: {exc}"
            self._step(f"[error] {result.error}")
            self._set_state(_state.FAILED, result.error)
            return result
        except Exception as exc:  # defensive: the task engine must not crash the CLI
            result.status = "fatal"
            result.error = f"Task engine error: {exc}"
            self._step(f"[error] {result.error}")
            self._set_state(_state.FAILED, result.error)
            return result

    def _plan_objective(self, objective: str, store) -> dict:
        """Ask the model to decompose ``objective`` into a task graph."""
        from .planner import project_intelligence

        intelligence = project_intelligence(store, objective)
        from .planner import planner_prompt

        prompt = planner_prompt(objective, intelligence)
        raw = _resilient_chat(
            self.client,
            [{"role": "user", "content": prompt}],
            format="json",
            should_stop=self.should_stop,
        )
        from .planner import parse_tasks

        parsed = _chat_value(raw)
        specs = parse_tasks(parsed)
        if not specs:
            specs = [
                {
                    "title": objective,
                    "description": objective,
                    "kind": "implement",
                    "verification": ["confirm the objective is satisfied"],
                }
            ]
        from .tasks import build_graph_from_specs, chunk_graph

        try:
            graph = build_graph_from_specs(specs)
            if any(t.complexity == "large" for t in graph.tasks.values()):
                graph = chunk_graph(graph)
            # Guarantee verification on every task via planner heuristics.
            from .planner import _derive_verification

            for task in graph.tasks.values():
                if not task.verification:
                    task.verification = _derive_verification(
                        {"title": task.title, "description": task.description,
                         "kind": task.kind, "files": task.files},
                        intelligence,
                    )
            graph.recompute_statuses()
            graph.validate()
        except Exception as exc:  # noqa: BLE001 - recover from a malformed plan
            # The model produced an invalid dependency graph (unknown task ref
            # or a cycle). Fall back to a single, honest review-style task so
            # execution still proceeds instead of hard-failing the whole run.
            self._step(f"[planner] malformed task graph, falling back: {exc}")
            graph = build_graph_from_specs(
                [
                    {
                        "id": "task-1",
                        "title": f"Implement and verify: {objective}",
                        "description": objective,
                        "kind": "implement",
                        "verification": ["verify the requested changes behave as intended"],
                    }
                ]
            )
        return {"graph": graph, "text": plan_text(graph), "count": len(graph)}

    def _build_executor(self, store) -> TaskExecutor:
        """Build a TaskExecutor with SAFE/PLAN mode gating wired in."""
        from .context import git_status as _git_status

        # Capture a dirty baseline before execution begins (Phase 4.4).
        dirty_raw = _git_status(self.ws.root)
        git_baseline: set[str] = set()
        for line in dirty_raw.splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                path = parts[1].strip()
                if path and not path.startswith(".ascs"):
                    git_baseline.add(path)

        def wired_run(executor, task):
            emit_task_started(self.event_sink, task.id, task.title)
            outcome = executor._default_run_task(executor, task)
            emit_task_completed(
                self.event_sink, task.id, ok=outcome.ok, summary=outcome.summary
            )
            return outcome

        return TaskExecutor(
            config=self.config,
            client=self.client,
            workspace=self.ws,
            store=store,
            event_sink=self.event_sink,
            log=self.log,
            should_stop=self.should_stop,
            run_task=wired_run,
            approver=self.approver,
            git_baseline=git_baseline,
        )

    # -- internals ----------------------------------------------------------

    def _run_tool(self, tool: str, arguments: dict[str, Any], iteration: int) -> ToolResult:
        """Emit lifecycle/command/file events around a single tool call."""
        target = None
        for key in ("path", "command"):
            val = arguments.get(key)
            if isinstance(val, str) and val:
                target = val
                break

        emit_tool_started(self.event_sink, tool, target)
        started = _time.monotonic()

        is_command = tool == "run_command"
        if is_command:
            self._emit_command("start", arguments.get("command", ""))

        if self.config.is_safe_mode and tool in MODIFY_TOOLS:
            display_args = _json.dumps(sorted(arguments.items()))
            if not self.approver(f"{tool} {display_args}"):
                self._step(f"[{iteration:02d}] SAFE mode: action declined by operator.")
                self._set_state(_state.PLANNING, "Action declined by operator")
                result = ToolResult(
                    tool,
                    "Operator declined the action in SAFE mode.",
                    ok=False,
                )
            else:
                result = execute_tool(tool, arguments, self.ws, self.config)
        else:
            result = execute_tool(tool, arguments, self.ws, self.config)

        elapsed = _time.monotonic() - started

        if is_command:
            cmd = arguments.get("command", "")
            exit_code = self._rc_from_note(result.note)
            self._emit_command("end", cmd, exit_code=exit_code, elapsed=elapsed)
            preview = result.output
            if len(preview) > 6000:
                preview = preview[:5997] + "..."
            emit_command_output(
                self.event_sink,
                cmd,
                preview,
                exit_code=exit_code,
                ok=(exit_code or 0) == 0,
            )

        if tool == "set_plan" and result.ok:
            plan = Plan.from_value(arguments)
            self._plan = plan
            result_note = "plan recorded"
            self._step(f"[{iteration:02d}] Plan recorded ({len(plan.steps)} steps).")
            self._set_state(_state.PLANNING, "Plan recorded")
            result = ToolResult(tool, plan.to_text(), note=result_note)

        if tool in ("write_file", "apply_patch", "delete_file", "move_file", "copy_file", "run_command"):
            if tool != "run_command":
                emit_file_event(
                    self.event_sink,
                    FILE_EVENT_FOR_TOOL.get(tool, "file_written"),
                    target or "",
                    ok=result.ok,
                )
            if tool != "set_plan":
                if is_test_command(arguments.get("command", "")):
                    self._set_state(_state.VERIFYING, "Running tests / verification")
                else:
                    self._set_state(_state.EXECUTING, f"Tool {tool} executed")

        self._step(f"[{iteration:02d}] {self._result_line(tool, result)}")
        if self.config.verbose and result.ok:
            self.log(self._preview_output(result))

        emit_tool_completed(self.event_sink, tool, ok=result.ok, target=target, elapsed=elapsed)
        return result

    @staticmethod
    def _rc_from_note(note: str) -> int:
        if note and "timed out" in note:
            return -1  # non-zero sentinel: a timeout is never success
        m = re.search(r"exit code (\d+)", note or "")
        return int(m.group(1)) if m else 0

    def _bump_malformed(self, iteration: int) -> bool:
        """Count a malformed/unusable reply; True when the retry limit is hit."""
        self._malformed_count += 1
        return self._malformed_count >= self.config.malformed_retry_limit

    def _messages_for_request(self) -> list[dict[str, str]]:
        messages = list(self._messages)
        budget = self.config.context_budget_chars
        total = sum(len(_content(m)) for m in messages)
        # Never trim the system prompt (index 0) or the user task (index 1).
        while total > budget and len(messages) > 3:
            removed = messages.pop(2)
            total -= len(_content(removed))
        return messages

    def _is_repeated_call(self, tool: str, arguments: dict[str, Any]) -> bool:
        key = (tool, tuple(sorted(arguments.items())))
        if self._last_call == key:
            return True
        return False

    def _result_line(self, tool: str, result: ToolResult) -> str:
        head = result.note or ("succeeded" if result.ok else "FAILED")
        return f"[result] {tool}: {head}"

    def _preview_output(self, result: ToolResult) -> str:
        body = result.output
        if len(body) > 600:
            body = body[:597] + "..."
        return f"[output] {result.name}: {body}"

    def _finish(
        self,
        status: str,
        summary: str,
        iterations: int,
        terminal_state: str,
        error: str = "",
        plan: Plan | None = None,
    ) -> LoopResult:
        self.tracker.finish(terminal_state, summary)
        if terminal_state == _state.COMPLETE:
            emit_completed(self.event_sink, summary, summary=summary)
        elif terminal_state == _state.CANCELLED:
            emit_stopped(self.event_sink, summary)
        else:
            emit_error(self.event_sink, summary, error or summary)
        return LoopResult(
            status=status,
            state=terminal_state,
            summary=summary,
            iterations=iterations,
            steps=list(self._steps),
            error=error,
            plan=plan,
        )


def run_agent(
    config: AgentConfig,
    client: OllamaClient,
    task: str,
    *,
    approver: Callable[[str], bool] | None = None,
    log: Callable[[str], None] | None = None,
    event_sink: EventSink | None = None,
    should_stop: Callable[[], bool] | None = None,
    tracker: StateTracker | None = None,
) -> LoopResult:
    workspace = Workspace(config.workspace)
    loop = AgentLoop(
        config,
        client,
        workspace,
        approver=approver,
        log=log,
        event_sink=event_sink,
        should_stop=should_stop,
        tracker=tracker,
    )
    return loop.run(task)


def run_graph_agent(
    config: AgentConfig,
    client: OllamaClient,
    objective: str,
    *,
    log: Callable[[str], None] | None = None,
    event_sink: EventSink | None = None,
    should_stop: Callable[[], bool] | None = None,
    tracker: StateTracker | None = None,
    resume: bool = False,
) -> GraphLoopResult:
    """Run the task-graph-driven pipeline end-to-end (plan -> execute -> verify)."""
    workspace = Workspace(config.workspace)
    loop = AgentLoop(
        config,
        client,
        workspace,
        log=log,
        event_sink=event_sink,
        should_stop=should_stop,
        tracker=tracker,
    )
    return loop.run_graph(objective, resume=False if not resume else True)