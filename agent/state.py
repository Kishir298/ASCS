"""Explicit agent lifecycle states.

The execution engine moves through a small, explicit state machine so that the
current state is always visible (to the UI, CLI, and any future ASIS/TIVISS
integration) and a failed/unfinished task is never silently dropped.

Lifecycle (happy path):
    IDLE -> RECEIVING_TASK -> PLANNING -> EXECUTING -> VERIFYING -> COMPLETE

Terminal/abort states:
    FAILED     - a fatal error ended the run.
    CANCELLED  - the operator (or an interrupt) stopped the run.
    TIMEOUT    - iteration/time budget exhausted before completion.
"""

from __future__ import annotations

import threading
import time as _time
from dataclasses import dataclass, field

# Canonical state names.
IDLE = "idle"
RECEIVING_TASK = "receiving_task"
PLANNING = "planning"
EXECUTING = "executing"
VERIFYING = "verifying"
COMPLETE = "complete"
FAILED = "failed"
CANCELLED = "cancelled"
TIMEOUT = "timeout"

ALL_STATES = (
    IDLE,
    RECEIVING_TASK,
    PLANNING,
    EXECUTING,
    VERIFYING,
    COMPLETE,
    FAILED,
    CANCELLED,
    TIMEOUT,
)

# Body states (not IDLE/terminal) used for progress reporting.
ACTIVE_STATES = (RECEIVING_TASK, PLANNING, EXECUTING, VERIFYING)
TERMINAL_STATES = (COMPLETE, FAILED, CANCELLED, TIMEOUT)

# Colors / display labels.
STATE_LABELS = {
    IDLE: "IDLE",
    RECEIVING_TASK: "RECEIVING TASK",
    PLANNING: "PLANNING",
    EXECUTING: "EXECUTING",
    VERIFYING: "VERIFYING",
    COMPLETE: "COMPLETE",
    FAILED: "FAILED",
    CANCELLED: "CANCELLED",
    TIMEOUT: "TIMEOUT",
}


@dataclass
class StateSnapshot:
    """Readable state of a run."""

    state: str = IDLE
    mode: str = "AUTO"
    task: str = ""
    started_at: float | None = None
    ended_at: float | None = None
    transitions: list[tuple[str, float]] = field(default_factory=list)
    message: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def elapsed(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at is not None else _time.time()
        return end - self.started_at

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "label": STATE_LABELS.get(self.state, self.state.upper()),
            "mode": self.mode,
            "task": self.task,
            "active": self.is_active,
            "terminal": self.is_terminal,
            "message": self.message,
            "elapsed": self.elapsed,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


def is_valid_state(state: str) -> bool:
    return state in ALL_STATES


class StateTracker:
    """Small thread-safe state holder shared by the loop, worker, and UI."""

    def __init__(self, initial: str = IDLE) -> None:
        if not is_valid_state(initial):
            raise ValueError(f"Unknown state {initial!r}")
        self._lock = threading.Lock()
        self._snapshot = StateSnapshot(state=initial)
        self._handlers: list = []  # callables(state: str, prev: str)

    @property
    def state(self) -> str:
        with self._lock:
            return self._snapshot.state

    @property
    def snapshot(self) -> StateSnapshot:
        with self._lock:
            snap = self._snapshot
            return StateSnapshot(
                state=snap.state,
                mode=snap.mode,
                task=snap.task,
                started_at=snap.started_at,
                ended_at=snap.ended_at,
                transitions=list(snap.transitions),
                message=snap.message,
            )

    def configure(self, *, mode: str, task: str = "") -> None:
        with self._lock:
            self._snapshot.mode = mode
            self._snapshot.task = task

    def start(self, state: str = RECEIVING_TASK, message: str = "") -> None:
        """Begin a new run: reset elapsed timing and transition history.

        Each call to ``start`` marks a fresh run so a second task in a long-lived
        process (e.g. the web UI) does not inherit the previous run's
        ``started_at``/transitions. ``IDLE -> RECEIVING_TASK`` counts as the
        first entry into the run.
        """
        if not is_valid_state(state):
            raise ValueError(f"Unknown state {state!r}")
        with self._lock:
            prev = self._snapshot.state
            self._snapshot.transitions = []
            self._snapshot.started_at = _time.time()
            self._snapshot.ended_at = None
            self._snapshot.message = message
            if prev != state:
                self._snapshot.transitions.append((state, _time.time()))
                self._snapshot.state = state
        if prev != state:
            for handler in list(self._handlers):
                try:
                    handler(state, prev)
                except Exception:  # pragma: no cover - observers must not break core
                    pass

    def finish(self, state: str, message: str = "") -> None:
        if not is_valid_state(state):
            raise ValueError(f"Unknown state {state!r}")
        with self._lock:
            prev = self._snapshot.state
            self._snapshot.ended_at = _time.time()
            self._snapshot.message = message
            # Preserve the substantive body state (FAILED/CANCELLED/TIMEOUT/
            # COMPLETE) as a named transition for history clarity.
            self._snapshot.transitions.append((state, _time.time()))
            self._snapshot.state = state
        if prev != state:
            for handler in list(self._handlers):
                try:
                    handler(state, prev)
                except Exception:  # pragma: no cover - observers must not break core
                    pass

    def set(self, state: str, message: str = "") -> None:
        if not is_valid_state(state):
            raise ValueError(f"Unknown state {state!r}")
        with self._lock:
            prev = self._snapshot.state
            if prev != state:
                if self._snapshot.started_at is not None:
                    self._snapshot.transitions.append((state, _time.time()))
                self._snapshot.state = state
            if message:
                self._snapshot.message = message
        if prev != state:
            for handler in list(self._handlers):
                try:
                    handler(state, prev)
                except Exception:  # pragma: no cover - observers must not break core
                    pass

    def on_transition(self, handler) -> None:
        """Register ``handler(state, prev)``; called on every state change."""
        self._handlers.append(handler)

    def reset(self) -> None:
        with self._lock:
            self._snapshot = StateSnapshot(state=IDLE)


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