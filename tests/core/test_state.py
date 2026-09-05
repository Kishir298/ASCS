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


def test_start_begins_fresh_run_after_terminal(tmp_path):
    # A second task in a long-lived process must not inherit the previous
    # run's timing or transition history (web UI successive runs).
    import time as _time

    t = StateTracker(IDLE)
    t.configure(mode="AUTO", task="first")
    t.start(RECEIVING_TASK)
    t.set(EXECUTING)
    t.finish(COMPLETE, "first done")
    first_started = t.snapshot.started_at
    assert len(t.snapshot.transitions) >= 2

    _time.sleep(0.05)  # let the clock tick past the first run's timestamp
    t.configure(mode="AUTO", task="second")
    t.start(RECEIVING_TASK)
    snap = t.snapshot
    assert snap.started_at is not None
    assert snap.started_at > first_started
    assert snap.ended_at is None
    # Old run's transitions are gone; only the fresh RECEIVING entry remains.
    # The recorded transition timestamp and started_at are captured within the
    # same lock but at different instants, so compare with a tolerance.
    assert len(snap.transitions) == 1
    assert snap.transitions[0][0] == RECEIVING_TASK
    assert snap.transitions[0][1] == pytest.approx(snap.started_at, abs=1e-3)
    assert snap.state == RECEIVING_TASK


def test_start_from_idle_counts_entry_transition():
    t = StateTracker(IDLE)
    t.start(RECEIVING_TASK)
    assert t.snapshot.transitions[0][0] == RECEIVING_TASK
    assert t.snapshot.transitions[0][1] == pytest.approx(
        t.snapshot.started_at, abs=1e-3
    )


def test_labels_are_upper_ascii():
    for state, label in STATE_LABELS.items():
        assert is_valid_state(state)
        assert label.isupper()
        assert " " not in label[0]  # no leading-space labels