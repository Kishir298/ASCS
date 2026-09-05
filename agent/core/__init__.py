"""A.S.C.S. core orchestration.

Central runtime: :class:`AgentLoop`, lifecycle/state orchestration,
cancellation, high-level coordination.

Canonical implementation lives here (``agent/core/loop.py``,
``agent/core/state.py``). Compatibility shims at ``agent/loop.py`` and
``agent/state.py`` re-export these so existing ``from agent.loop import …``
imports keep working. New code should import from ``agent.core`` submodules.

Re-exports are lazy (PEP 562) to avoid circular imports between domains::

    from agent.core.loop import AgentLoop
    from agent.core import AgentLoop  # lazy, same object
"""

from __future__ import annotations

_LAZY: dict[str, str] = {
    "AgentLoop": "agent.core.loop",
    # Phase 1 brain: intent/decision layer.
    "Decision": "agent.core.intent",
    "classify_request": "agent.core.intent",
    "INTENT_CATEGORIES": "agent.core.intent",
    "GraphLoopResult": "agent.core.loop",
    "LoopResult": "agent.core.loop",
    "run_agent": "agent.core.loop",
    "run_graph_agent": "agent.core.loop",
    "StateTracker": "agent.core.state",
    "StateSnapshot": "agent.core.state",
    "is_valid_state": "agent.core.state",
    "IDLE": "agent.core.state",
    "RECEIVING_TASK": "agent.core.state",
    "PLANNING": "agent.core.state",
    "EXECUTING": "agent.core.state",
    "VERIFYING": "agent.core.state",
    "COMPLETE": "agent.core.state",
    "FAILED": "agent.core.state",
    "CANCELLED": "agent.core.state",
    "TIMEOUT": "agent.core.state",
    "ALL_STATES": "agent.core.state",
    "ACTIVE_STATES": "agent.core.state",
    "TERMINAL_STATES": "agent.core.state",
    "STATE_LABELS": "agent.core.state",
    "AgentEvent": "agent.events",
    "EventSink": "agent.events",
    "null_sink": "agent.events",
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
