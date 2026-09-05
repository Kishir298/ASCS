"""Compatibility shim: canonical implementation lives at ``agent.core.state``.

Preserved so existing ``from agent.state import …`` imports keep working
after the Phase 0 move. New code should import from ``agent.core``.
"""

from __future__ import annotations

from agent.core.state import *  # noqa: F401,F403
from agent.core.state import (
    ACTIVE_STATES,
    ALL_STATES,
    CANCELLED,
    COMPLETE,
    EXECUTING,
    FAILED,
    IDLE,
    PLANNING,
    RECEIVING_TASK,
    STATE_LABELS,
    TERMINAL_STATES,
    TIMEOUT,
    VERIFYING,
    StateSnapshot,
    StateTracker,
    is_valid_state,
)

__all__ = [
    "IDLE",
    "RECEIVING_TASK",
    "PLANNING",
    "EXECUTING",
    "VERIFYING",
    "COMPLETE",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "ALL_STATES",
    "ACTIVE_STATES",
    "TERMINAL_STATES",
    "STATE_LABELS",
    "StateSnapshot",
    "StateTracker",
    "is_valid_state",
]
