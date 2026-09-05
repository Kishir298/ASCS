"""Tests for the verification boundary (``agent.verification``).

Phase 0 scope: verification logic is intentionally distributed (executor
verify/retry/cascade, toolchain-derived acceptance, loop guards, verification
events). These tests pin the re-export contract of the architectural home so
Phase 6 can harden the flow without moving addresses underneath callers.
No verification behavior is created or changed here.
"""

from __future__ import annotations

import agent.verification as verification


def test_verification_package_reexports_executor_types():
    assert verification.TaskOutcome is not None
    assert verification.VerificationResult is not None
    assert verification.TaskActionLog is not None


def test_verification_reexports_are_executor_canonical():
    from agent.execution import executor

    assert verification.TaskOutcome is executor.TaskOutcome
    assert verification.VerificationResult is executor.VerificationResult
    assert verification.TaskActionLog is executor.TaskActionLog


def test_verification_unknown_attribute_raises():
    try:
        verification.NonexistentVerificationThing  # type: ignore[attr-defined]
    except AttributeError:
        pass
    else:  # pragma: no cover - only on contract regression
        raise AssertionError("expected AttributeError for unknown attribute")
