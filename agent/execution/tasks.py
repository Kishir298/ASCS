"""A.S.C.S. task engine (task graph model).

The master plan moves from 'one large prompt' to structured, resumable work:

    Task  --(dependencies)--> TaskGraph --> execution order -> per-task state

Each :class:`Task` carries everything needed to execute and verify one bounded
unit of work. A :class:`TaskGraph` orders tasks by dependency and tracks
per-task status so a run can be paused, resumed and inspected.

This module is the *model* used by the planning engine (Phase 3). The
execution engine (later phases) consumes the same graph.

States
------
``PENDING`` (not ready) -> ``READY`` (dependencies satisfied) -> ``RUNNING``
-> ``COMPLETED`` / ``FAILED`` / ``CANCELLED`` / ``SKIPPED``. A task whose
dependency failed becomes ``BLOCKED``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from agent.context.index import DEFAULT_STATE_DIR, ContextError
from agent.models import Plan

PENDING = "pending"
READY = "ready"
RUNNING = "running"
BLOCKED = "blocked"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
SKIPPED = "skipped"

VALID_STATUSES = frozenset(
    {PENDING, READY, RUNNING, BLOCKED, COMPLETED, FAILED, CANCELLED, SKIPPED}
)
TERMINAL_STATUSES = frozenset({COMPLETED, FAILED, CANCELLED, SKIPPED})


@dataclass
class Task:
    """One bounded unit of work with its own acceptance criteria."""

    id: str
    title: str
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    priority: int = 0  # higher runs first among ready tasks
    status: str = PENDING
    files: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    retry_count: int = 0
    failure_reason: str = ""
    result: str = ""
    complexity: str = "medium"  # small | medium | large
    kind: str = "implement"  # plan | inspect | implement | verify | review
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Task":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})

    def touch(self) -> None:
        self.updated_at = time.time()

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_small(self) -> bool:
        return self.complexity in ("", "small")


class TaskGraphError(ContextError):
    """Raised for invalid task graph operations."""


class TaskGraph:
    """Ordered, dependency-aware collection of tasks."""

    def __init__(self, tasks: Iterable[Task] | None = None) -> None:
        self.tasks: dict[str, Task] = {}
        if tasks:
            for task in tasks:
                self.add(task)

    # -- construction -----------------------------------------------------

    def add(self, task: Task) -> None:
        if task.id in self.tasks:
            raise TaskGraphError(f"Duplicate task id: {task.id!r}")
        self.tasks[task.id] = task

    def task(self, task_id: str) -> Task:
        try:
            return self.tasks[task_id]
        except KeyError:
            raise TaskGraphError(f"Unknown task id: {task_id!r}") from None

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self.ordered())

    # -- dependency handling -----------------------------------------------

    def validate(self) -> None:
        """Raise if a task depends on an unknown id or a cycle exists.

        Returns ``None`` on success (empty graph is valid).
        """
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    raise TaskGraphError(
                        f"Task {task.id!r} depends on unknown task {dep!r}"
                    )
        # Cycle detection via three-colour DFS.
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise TaskGraphError(f"Dependency cycle involving task {task_id!r}")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dep in self.tasks[task_id].dependencies:
                visit(dep)
            visiting.discard(task_id)
            visited.add(task_id)

        for task_id in self.tasks:
            visit(task_id)

    def _dependencies_satisfied(self, task: Task) -> bool:
        return all(
            self.tasks[dep].status == COMPLETED for dep in task.dependencies
        )

    def _dependencies_blocked(self, task: Task) -> bool:
        return any(
            self.tasks[dep].status in (FAILED, CANCELLED, SKIPPED, BLOCKED)
            for dep in task.dependencies
        )

    def ordered(self) -> list[Task]:
        """Topologically ordered tasks (dependencies first)."""
        self.validate()
        visited: set[str] = set()
        result: list[Task] = []

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            visited.add(task_id)
            for dep in self.tasks[task_id].dependencies:
                visit(dep)
            result.append(self.tasks[task_id])

        for task_id in sorted(self.tasks):
            visit(task_id)
        return result

    # -- state ------------------------------------------------------------

    def ready_tasks(self) -> list[Task]:
        """Tasks that can run now (READY or PENDING with satisfied deps, not terminal)."""
        candidates = [
            t
            for t in self.tasks.values()
            if t.status in (PENDING, READY) and not t.is_terminal
        ]
        ready: list[Task] = []
        for task in candidates:
            if self._dependencies_blocked(task):
                continue  # handled by recompute_statuses
            if self._dependencies_satisfied(task):
                ready.append(task)
        ready.sort(key=lambda t: (-t.priority, t.id))
        return ready

    def recompute_statuses(self) -> None:
        """Refresh task statuses from dependency state (readiness/blocking)."""
        for task in self.tasks.values():
            if task.status in (PENDING, READY):
                if self._dependencies_blocked(task):
                    task.status = BLOCKED
                    task.failure_reason = "a dependency failed"
                elif self._dependencies_satisfied(task):
                    if task.status == PENDING:
                        task.status = READY
                else:
                    task.status = PENDING
        self.validate()

    def mark(self, task_id: str, status: str, **kwargs) -> Task:
        """Update a task's status (validating against the state machine)."""
        if status not in VALID_STATUSES:
            raise TaskGraphError(f"Invalid status {status!r}")
        task = self.task(task_id)
        old = task.status
        task.status = status
        for key, value in kwargs.items():
            if key in {"retry_count", "failure_reason", "result"}:
                setattr(task, key, value)
        if status == FAILED and not task.failure_reason:
            task.failure_reason = "task failed"
        task.touch()
        if old != status:
            self.recompute_statuses()
        return task

    def progress(self) -> dict:
        counts: dict[str, int] = {s: 0 for s in VALID_STATUSES}
        for task in self.tasks.values():
            if task.status in counts:
                counts[task.status] += 1
        return {
            "total": len(self.tasks),
            "completed": counts[COMPLETED],
            "failed": counts[FAILED],
            "running": counts[RUNNING],
            "blocked": counts[BLOCKED],
            "pending": counts[PENDING] + counts[READY],
            "cancelled": counts[CANCELLED],
            "skipped": counts[SKIPPED],
        }

    @property
    def all_complete(self) -> bool:
        return bool(self.tasks) and all(
            t.status in (COMPLETED, SKIPPED) for t in self.tasks.values()
        )

    def tasks_by_status(self, status: str) -> list[Task]:
        """Tasks currently in ``status``, ordered by id for determinism."""
        return [
            self.tasks[tid]
            for tid in sorted(self.tasks)
            if self.tasks[tid].status == status
        ]

    @property
    def failed(self) -> list[Task]:
        return self.tasks_by_status(FAILED)

    @property
    def blocked(self) -> list[Task]:
        return self.tasks_by_status(BLOCKED)

    @property
    def first_failure(self) -> Task | None:
        order = {t.id: i for i, t in enumerate(self.ordered())}
        seen: Task | None = None
        for task in self.failed:
            if seen is None or order.get(task.id, 0) < order.get(seen.id, 0):
                seen = task
        return seen

    def cascade_cancel(self) -> list[str]:
        """Mark as CANCELLED every non-terminal task that cannot proceed.

        A task is cancelled here only when it is *blocked* by a failed/cancelled
        dependency, or is itself unstarted and transitively depends on a
        terminal-failure path. Independent unscheduled tasks are left alone so
        they can still run.
        """
        self.recompute_statuses()
        cancelled: list[str] = []
        for task in list(self.tasks.values()):
            if task.status == BLOCKED:
                cancelled.append(task.id)
        for task_id in cancelled:
            self.mark(task_id, CANCELLED)
        return cancelled

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "updated_at": time.time(),
            "tasks": [t.to_dict() for t in sorted(self.tasks.values(), key=lambda t: t.id)],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TaskGraph":
        graph = cls()
        for raw in payload.get("tasks", []):
            if isinstance(raw, dict):
                graph.add(Task.from_dict(raw))
        return graph


def plan_to_graph(
    plan: Plan,
    *,
    goal: str = "",
    prefix: str = "task",
) -> TaskGraph:
    """Convert a :class:`~agent.models.Plan` into a sequential :class:`TaskGraph`.

    Each plan step becomes a task. Steps are ordered sequentially (each step
    depends on the previous one), which is the natural reading of a `set_plan`
    output for a plain ordered list.
    """
    graph = TaskGraph()
    previous: str | None = None
    for index, step in enumerate(plan.steps):
        task = Task(
            id=f"{prefix}-{index + 1}",
            title=str(step),
            description=str(step),
            dependencies=[previous] if previous else [],
            status=READY if previous is None else PENDING,
        )
        graph.add(task)
        previous = task.id
    return graph


def build_graph_from_specs(
    specs: Sequence[Task | dict],
    *,
    prefix: str = "task",
    suppress_status_recompute: bool = False,
) -> TaskGraph:
    """Build a dependency-aware :class:`TaskGraph` from structured task specs.

    Each spec is either a :class:`Task` or a dict with keys matching the
    ``Task`` dataclass fields (``id``, ``title``, ``description``,
    ``dependencies``, ``files``, ``commands``, ``verification``,
    ``complexity``, ``kind``, ``priority``, ...).

    Unlike :func:`plan_to_graph` (which wires a flat list into a linear chain),
    this preserves the **explicit dependency graph** declared by each spec, so
    the result can express fan-in/fan-out (the ``T1 -> T2/T3 -> T4`` shape).

    Missing ids default to ``prefix-N`` deterministically. After construction
    the graph's dependency graph is validated (cycle + unknown-dependency
    detection).
    """
    graph = TaskGraph()
    for index, spec in enumerate(specs, start=1):
        if isinstance(spec, Task):
            task = spec
        elif isinstance(spec, dict):
            payload = dict(spec)
            if not payload.get("id"):
                payload["id"] = f"{prefix}-{index}"
            task = Task.from_dict(payload)
        else:
            raise TaskGraphError(
                f"Invalid task spec #{index}: expected Task or dict, got {type(spec).__name__}"
            )
        if not task.title:
            task.title = task.description or task.id
        graph.add(task)

    graph.validate()
    if not suppress_status_recompute:
        graph.recompute_statuses()
    return graph


def chunk_graph(
    graph: TaskGraph,
    *,
    split_at_complexity: str = "large",
    max_files_per_task: int = 8,
) -> TaskGraph:
    """Automatically split oversized tasks into smaller, coherent subtasks.

    * A task whose ``complexity`` reaches ``split_at_complexity`` is split
      along its declared files: one subtask per file (plus one for any
      commands), each inheriting the parent's dependencies and a new dependency
      on its siblings so they stay ordered.
    * Subtasks too small to be meaningful (no files, no commands, no
      verification, ``small`` complexity) are left as-is rather than over-split.

    Returns a *new* graph so the caller keeps the original intact. The IDs of
    subtasks are ``<parent>.<n>``. This is a deterministic, heuristic pass used
    by the planner before execution.
    """
    result = TaskGraph()

    for task in sorted(graph.tasks.values(), key=lambda t: t.id):
        if task.complexity != split_at_complexity:
            result.add(task)
            continue

        pieces: list[tuple[str, str, list[str]]] = []
        for index, file in enumerate(task.files[:max_files_per_task], start=1):
            pieces.append(
                (f"{task.id}.{index}", f"{task.title}: {file}", [file])
            )
        if not pieces:
            pieces.append((f"{task.id}.1", task.title, []))
        if task.commands and len(pieces) < max_files_per_task + 1:
            pieces.append(
                (
                    f"{task.id}.{len(pieces) + 1}",
                    f"{task.title}: run commands",
                    [],
                )
            )

        for index, (sub_id, sub_title, files) in enumerate(pieces, start=1):
            deps = list(task.dependencies)
            if len(pieces) > 1 and index > 1:
                deps.append(pieces[index - 2][0])
            result.add(
                Task(
                    id=sub_id,
                    title=sub_title,
                    description=task.description,
                    dependencies=deps,
                    priority=task.priority,
                    files=files,
                    commands=list(task.commands),
                    verification=list(task.verification),
                    complexity="medium",
                    kind=task.kind,
                    status=PENDING,
                )
            )
    result.recompute_statuses()
    result.validate()
    return result


__all__ = [
    "BLOCKED",
    "CANCELLED",
    "COMPLETED",
    "FAILED",
    "PENDING",
    "READY",
    "RUNNING",
    "SKIPPED",
    "TERMINAL_STATUSES",
    "Task",
    "TaskGraph",
    "TaskGraphError",
    "VALID_STATUSES",
    "build_graph_from_specs",
    "chunk_graph",
    "plan_to_graph",
]