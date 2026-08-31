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

from .context import DEFAULT_STATE_DIR, ContextError
from .models import Plan

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
    depends on the previous one) when the plan is a plain ordered list, which
    is the natural reading of a `set_plan` output.
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
    "plan_to_graph",
]