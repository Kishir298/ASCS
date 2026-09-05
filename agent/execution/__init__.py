"""A.S.C.S. execution.

Task-graph structures and execution: ready-task scheduling, task-scoped
prompts, verification gates, bounded retries, cascade-cancel, persistence.

Canonical implementation: ``agent/execution/tasks.py``,
``agent/execution/executor.py``. Shims at ``agent/tasks.py`` and
``agent/executor.py`` preserve old imports. The executor is not the brain —
orchestration stays in ``agent.core``.

Re-exports here are lazy (PEP 562) so importing the package never triggers
circular imports between domains. Both styles work::

    from agent.execution.tasks import TaskGraph
    from agent.execution import TaskGraph  # lazy, same object
"""

from __future__ import annotations

_LAZY: dict[str, str] = {
    "Task": "agent.execution.tasks",
    "TaskGraph": "agent.execution.tasks",
    "TaskExecutor": "agent.execution.executor",
    "TaskExecution": "agent.execution.executor",
    "TaskOutcome": "agent.execution.executor",
    "TaskActionLog": "agent.execution.executor",
    "VerificationResult": "agent.execution.executor",
    "PENDING": "agent.execution.tasks",
    "READY": "agent.execution.tasks",
    "RUNNING": "agent.execution.tasks",
    "COMPLETED": "agent.execution.tasks",
    "FAILED": "agent.execution.tasks",
    "CANCELLED": "agent.execution.tasks",
    "SKIPPED": "agent.execution.tasks",
    "build_graph_from_specs": "agent.execution.tasks",
    "chunk_graph": "agent.execution.tasks",
    "plan_to_graph": "agent.execution.tasks",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
