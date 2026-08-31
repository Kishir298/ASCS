"""Structured agent events for observability / UI consumption.

The agent loop emits :class:`AgentEvent` objects at real lifecycle points
(start, thinking, tool, command, file change, error, stop, done). Consumers
(the web UI or any future ASIS/TIVISS integration) subscribe via an
``event_sink`` callable and receive structured JSON rather than parsing
terminal strings.

Events carry concise *operational* status only. They NEVER expose the model's
private chain-of-thought; ``agent_thinking`` carries just a short human-readable
display phrase such as "Inspecting the test suite...".
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

# Full set of event types the loop may emit.
EVENT_TYPES = (
    "agent_started",
    "agent_thinking",
    "model_started",
    "model_completed",
    "tool_started",
    "tool_completed",
    "file_read",
    "file_written",
    "patch_applied",
    "command_started",
    "command_output",
    "command_completed",
    "test_started",
    "test_completed",
    "agent_error",
    "agent_stopped",
    "agent_completed",
    "mode_changed",
    "activity",
    "status",
    "task_plan",
    "task_created",
    "task_ready",
    "task_started",
    "task_blocked",
    "task_verified",
    "verification_started",
    "task_failed",
    "task_completed",
    "retry",
)


@dataclass(frozen=True)
class AgentEvent:
    """A structured, JSON-serializable event emitted by the agent loop."""

    type: str
    message: str = ""
    tool: str | None = None
    target: str | None = None
    command: str | None = None
    output: str | None = None
    exit_code: int | None = None
    ok: bool | None = None
    mode: str | None = None
    status: str | None = None
    summary: str | None = None
    error: str | None = None
    elapsed: float | None = None
    attempt: int | None = None
    retries_left: int | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Drop None fields to keep payloads small over the wire.
        return {k: v for k, v in data.items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def to_event_dict(event: AgentEvent) -> dict[str, Any]:
    """Return the JSON-serializable payload of an event."""
    return event.to_dict()


# A synchronous event sink; receives AgentEvent objects.
EventSink = Callable[[AgentEvent], None]


def null_sink(event: AgentEvent) -> None:
    """Default sink that discards events."""


def _emit(sink: EventSink | None, event: AgentEvent) -> None:
    """Deliver an event, swallowing sink errors so a broken consumer never
    crashes the agent loop."""
    if sink is None:
        return
    try:
        sink(event)
    except Exception:  # pragma: no cover - defensive; UI must not crash engine
        pass


# -- convenience emitters ----------------------------------------------------


def emit_started(sink: EventSink | None, message: str = "") -> None:
    _emit(sink, AgentEvent(type="agent_started", message=message or "Agent started"))


def emit_thinking(
    sink: EventSink | None,
    message: str = "",
    *,
    elapsed: float | None = None,
) -> None:
    _emit(
        sink,
        AgentEvent(
            type="agent_thinking",
            message=message or "Model is thinking...",
            elapsed=elapsed,
        ),
    )


def emit_model_started(sink: EventSink | None, message: str = "") -> None:
    _emit(
        sink,
        AgentEvent(
            type="model_started",
            message=message or "Model is thinking...",
        ),
    )


def emit_model_completed(
    sink: EventSink | None,
    message: str = "",
    *,
    elapsed: float | None = None,
) -> None:
    _emit(
        sink,
        AgentEvent(
            type="model_completed",
            message=message or "Model responded",
            elapsed=elapsed,
        ),
    )


def emit_command_output(
    sink: EventSink | None,
    command: str,
    output: str,
    *,
    exit_code: int | None = None,
    ok: bool = True,
) -> None:
    _emit(
        sink,
        AgentEvent(
            type="command_output",
            command=command,
            output=output,
            exit_code=exit_code,
            ok=ok,
            message=f"Command output for: {command}",
        ),
    )


def emit_tool_started(
    sink: EventSink | None, tool: str, target: str | None = None
) -> None:
    _emit(
        sink,
        AgentEvent(
            type="tool_started",
            tool=tool,
            target=target,
            message=f"Tool {tool} started",
        ),
    )


def emit_tool_completed(
    sink: EventSink | None,
    tool: str,
    *,
    ok: bool,
    target: str | None = None,
    elapsed: float | None = None,
) -> None:
    _emit(
        sink,
        AgentEvent(
            type="tool_completed",
            tool=tool,
            target=target,
            ok=ok,
            elapsed=elapsed,
            message=f"Tool {tool} {'completed' if ok else 'failed'}",
        ),
    )


def emit_status(sink: EventSink | None, status: str, message: str = "") -> None:
    _emit(
        sink,
        AgentEvent(type="status", status=status, message=message or status),
    )


def emit_activity(sink: EventSink | None, message: str) -> None:
    _emit(sink, AgentEvent(type="activity", message=message))


def emit_file_event(
    sink: EventSink | None,
    event_type: str,
    path: str,
    *,
    ok: bool = True,
) -> None:
    _emit(
        sink,
        AgentEvent(
            type=event_type,
            target=path,
            ok=ok,
            message=f"{event_type}: {path}",
        ),
    )


def emit_mode_changed(sink: EventSink | None, mode: str | None, message: str = "") -> None:
    _emit(
        sink,
        AgentEvent(
            type="mode_changed",
            mode=mode,
            message=message or f"Mode set to {mode}",
        ),
    )


def emit_test_started(sink: EventSink | None, command: str) -> None:
    _emit(
        sink,
        AgentEvent(
            type="test_started",
            command=command,
            message=f"Running tests: {command}",
        ),
    )


def emit_command_started(sink: EventSink | None, command: str) -> None:
    _emit(
        sink,
        AgentEvent(
            type="command_started",
            command=command,
            message=f"Running: {command}",
        ),
    )


def emit_command_completed(
    sink: EventSink | None,
    command: str,
    *,
    exit_code: int,
    ok: bool = True,
    elapsed: float | None = None,
) -> None:
    _emit(
        sink,
        AgentEvent(
            type="command_completed",
            command=command,
            exit_code=exit_code,
            ok=ok,
            elapsed=elapsed,
            message=f"Command exited with code {exit_code}",
        ),
    )


def emit_test_completed(
    sink: EventSink | None,
    command: str,
    *,
    exit_code: int,
    ok: bool = True,
    elapsed: float | None = None,
) -> None:
    _emit(
        sink,
        AgentEvent(
            type="test_completed",
            command=command,
            exit_code=exit_code,
            ok=ok,
            elapsed=elapsed,
            message=f"Tests {'passed' if ok else 'failed'} (exit {exit_code})",
        ),
    )


def emit_completed(
    sink: EventSink | None,
    message: str,
    *,
    summary: str = "",
    iter_count: int = 0,
) -> None:
    _emit(
        sink,
        AgentEvent(
            type="agent_completed",
            message=message,
            summary=summary or message,
        ),
    )


def emit_error(sink: EventSink | None, message: str, error: str = "") -> None:
    _emit(
        sink,
        AgentEvent(type="agent_error", message=message, error=error or message),
    )


def emit_stopped(sink: EventSink | None, message: str = "Agent stopped") -> None:
    _emit(sink, AgentEvent(type="agent_stopped", message=message))


def emit_task_plan(
    sink: EventSink | None,
    plan_text: str,
    *,
    task_count: int = 0,
) -> None:
    """Emit the rendered task plan for operator inspection."""
    _emit(
        sink,
        AgentEvent(
            type="task_plan",
            message=plan_text,
            summary=f"{task_count} task(s)",
        ),
    )


def emit_task_started(
    sink: EventSink | None,
    task_id: str,
    title: str,
) -> None:
    _emit(
        sink,
        AgentEvent(
            type="task_started",
            status=task_id,
            message=title or task_id,
        ),
    )


def emit_task_completed(
    sink: EventSink | None,
    task_id: str,
    *,
    ok: bool,
    summary: str = "",
) -> None:
    _emit(
        sink,
        AgentEvent(
            type="task_completed",
            status=task_id,
            ok=ok,
            summary=summary or "",
            message=f"Task {task_id} {'completed' if ok else 'failed'}",
        ),
    )


def emit_task_created(
    sink: EventSink | None,
    task_id: str,
    title: str,
    *,
    depends_on: tuple[str, ...] = (),
    n_files: int = 0,
) -> None:
    """Emit that the planner produced a task in the graph."""
    _emit(
        sink,
        AgentEvent(
            type="task_created",
            status=task_id,
            message=title or task_id,
            summary=f"{len(depends_on)} dep(s), {n_files} file(s)",
        ),
    )


def emit_task_ready(sink: EventSink | None, task_id: str, title: str) -> None:
    """Emit that a task's dependencies are satisfied and it is runnable."""
    _emit(
        sink,
        AgentEvent(
            type="task_ready",
            status=task_id,
            message=title or task_id,
        ),
    )


def emit_task_blocked(
    sink: EventSink | None,
    task_id: str,
    title: str,
    *,
    reason: str = "",
) -> None:
    """Emit that a task cannot run until its dependencies are satisfied."""
    _emit(
        sink,
        AgentEvent(
            type="task_blocked",
            status=task_id,
            message=title or task_id,
            error=reason or "dependency not satisfied",
        ),
    )


def emit_retry(
    sink: EventSink | None,
    *,
    task_id: str,
    attempt: int,
    retries_left: int,
    reason: str = "",
) -> None:
    """Emit a bounded retry of a task, verification, or model call."""
    _emit(
        sink,
        AgentEvent(
            type="retry",
            status=task_id,
            attempt=attempt,
            retries_left=retries_left,
            message=f"Retry {attempt} for task {task_id} "
            f"({retries_left} retr{'y' if retries_left == 1 else 'ies'} left)",
            error=reason or "",
        ),
    )


def emit_verification_started(
    sink: EventSink | None,
    task_id: str,
    *,
    attempt: int = 1,
) -> None:
    """Emit that a task is about to run its acceptance verification."""
    _emit(
        sink,
        AgentEvent(
            type="verification_started",
            status=task_id,
            attempt=attempt,
            message=f"Verifying task {task_id} (attempt {attempt})",
        ),
    )


def emit_task_verified(
    sink: EventSink | None,
    task_id: str,
    *,
    ok: bool,
    summary: str = "",
    attempt: int | None = None,
    retries_left: int | None = None,
) -> None:
    """Emit a per-task verification result (quality-gate outcome).

    ``attempt`` is the 1-based verification attempt that produced the result
    and ``retries_left`` the number of bounded retries that remain; both are
    optional structured fields (omitted for the default/single-shot path).
    """
    _emit(
        sink,
        AgentEvent(
            type="task_verified",
            status=task_id,
            ok=ok,
            message=f"Task {task_id} verify {'passed' if ok else 'failed'}",
            summary=summary or "",
            attempt=attempt,
            retries_left=retries_left,
        ),
    )


def emit_task_failed(
    sink: EventSink | None,
    task_id: str,
    reason: str,
) -> None:
    """Emit a per-task failure (model failure or failed verification)."""
    _emit(
        sink,
        AgentEvent(
            type="task_failed",
            status=task_id,
            ok=False,
            message=f"Task {task_id} failed: {reason}",
            error=reason,
        ),
    )
