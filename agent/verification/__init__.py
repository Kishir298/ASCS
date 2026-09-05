"""A.S.C.S. verification.

Architectural home for the verify → retry → fail/cascade flow. In Phase 0
verification logic remains distributed where it runs (no blind moves):

- ``agent.execution.executor``: ``_verify_task`` (every ``run …`` step must
  exit 0), ``max_verify_retries``, ``VerificationResult``/``TaskOutcome``,
  failure feedback, cascade-cancel via ``TaskGraph``.
- ``agent.context.toolchain``: evidence-based acceptance derivation
  (``_derive_verification`` in the planner).
- ``agent.core``: iteration/malformed/timeout guards; ``TIMEOUT``/``FAILED``.
- ``agent.events``: ``verification_started`` / ``retry`` / ``task_verified`` /
  ``task_failed`` with 1-based ``attempt`` + ``retries_left``.

This package re-exports the key verification types (lazy) so Phase 6 has a
single address. Phase 0 performs no reliability redesign.
"""

from __future__ import annotations

_LAZY: dict[str, str] = {
    "TaskOutcome": "agent.execution.executor",
    "VerificationResult": "agent.execution.executor",
    "TaskActionLog": "agent.execution.executor",
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
