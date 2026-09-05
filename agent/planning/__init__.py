"""A.S.C.S. planning.

Objective → validated task DAG: decomposition prompts, response parsing,
large-task chunking, verification guarantees.

Canonical implementation: ``agent/planning/planner.py``,
``agent/planning/prompts.py``. Shims at ``agent/planner.py`` and
``agent/prompts.py`` preserve old imports. Task execution lives in
``agent.execution`` — never moved here.

Re-exports are lazy (PEP 562) to avoid circular imports.
"""

from __future__ import annotations

_LAZY: dict[str, str] = {
    "COMPLEXITIES": "agent.planning.planner",
    "KINDS": "agent.planning.planner",
    "parse_tasks": "agent.planning.planner",
    "plan_objective": "agent.planning.planner",
    "plan_text": "agent.planning.planner",
    "planner_prompt": "agent.planning.planner",
    "project_intelligence": "agent.planning.planner",
    "system_prompt": "agent.planning.prompts",
    "task_message": "agent.planning.prompts",
    "malformed_feedback": "agent.planning.prompts",
    "tool_error_feedback": "agent.planning.prompts",
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
