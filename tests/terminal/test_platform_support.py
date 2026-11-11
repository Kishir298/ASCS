"""Regression tests for cross-platform runtime support."""

from __future__ import annotations

import os

from agent import doctor


def test_platform_check_passes_on_every_supported_host():
    """Windows and POSIX are both supported; only the shell differs."""
    result = doctor._check_runtime_platform()
    assert result.status == doctor.PASS
    if os.name == "nt":
        assert "Windows" in result.message
    else:
        assert "POSIX" in result.message
