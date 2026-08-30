"""Tests for the lifecycle state machine (agent.state)."""

from __future__ import annotations

import pytest

from agent.state import (
    CANCELLED,
    COMPLETE,
    EXECUTING,
    FAILED,
    IDLE,
    PLANNING,
    RECEIVING_TASK,
    TIMEOUT,
    VERIFYING,
    STATE_LABELS,
    StateSnapshot,
    StateTracker,
    is_valid_state,
    TERMINAL_STATES,
    ACTIVE_STATES,
)


def test_valid_states():
    for s in (IDLE, RECEIVING_TASK, PLANNING, EXECUTING, VERIFYING, COMPLETE, FAILED, CANCELLED, TIMEOUT):
        assert is_valid_state(s)
    assert not is_valid_state("bogus")


def test_terminal_and_active_partitions():
    assert COMPLETE in TERMINAL_STATES
    assert FAILED in TERMINAL_STATES
    assert CANCELLED in TERMINAL_STATES
    assert TIMEOUT in TERMINAL_STATES
    assert PLANNING in ACTIVE_STATES
    assert EXECUTING in ACTIVE_STATES


def test_invalid_initial_raises():
    with pytest.raises(ValueError):
        StateTracker("nope")


def test_basic_transitions():
    t = StateTracker(IDLE)
    assert t.state == IDLE
    t.configure(mode="AUTO", task="task")
    t.start(RECEIVING_TASK)
    t.set(PLANNING)
    snap = t.snapshot
    assert snap.state == PLANNING
    assert snap.mode == "AUTO"
    assert snap.task == "task"
    assert snap.elapsed is None or snap.elapsed >= 0
    t.finish(COMPLETE, "done")
    assert t.state == COMPLETE
    assert t.snapshot.is_terminal
    assert len(t.snapshot.transitions) >= 2


def test_snapshot_semantics():
    t = StateTracker(IDLE)
    snap = t.snapshot
    assert not snap.is_active
    assert not snap.is_terminal
    t.start(RECEIVING_TASK)
    assert t.snapshot.is_active
    t.finish(FAILED, "boom")
    assert t.snapshot.is_terminal
    assert t.snapshot.state == FAILED
    assert t.snapshot.message == "boom"
    assert t.snapshot.ended_at is not None


def test_transition_handlers_called():
    t = StateTracker(IDLE)
    seen = []
    t.on_transition(lambda state, prev: seen.append((prev, state)))
    t.start(RECEIVING_TASK)
    t.set(PLANNING)
    assert seen[-1] == (RECEIVING_TASK, PLANNING)


def test_reset_returns_idle():
    t = StateTracker()
    t.start(RECEIVING_TASK)
    assert t.state != IDLE
    t.reset()
    assert t.state == IDLE


def test_labels_are_upper_ascii():
    for state, label in STATE_LABELS.items():
        assert is_valid_state(state)
        assert label.isupper()
        assert " " not in label[0]  # no leading-space labels