"""Headless TUI entry tests: banner encoding + TTY guard (no live Ollama)."""

from __future__ import annotations

import os
import subprocess
import sys


def _venv_python() -> str:
    return sys.executable


def test_banner_survives_cp1252_headless():
    """Box-drawing banner must fall back to ASCII on legacy codepages."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"
    proc = subprocess.run(
        [_venv_python(), "-m", "agent.terminal"],
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    combined = proc.stdout + proc.stderr
    assert "UnicodeEncodeError" not in combined
    assert proc.returncode == 1  # TTY guard exits 1 headless
    assert "real terminal" in combined.lower()


def test_tty_guard_message_names_entry_point():
    """The headless message must point at a working launch command."""
    env = dict(os.environ)
    proc = subprocess.run(
        [_venv_python(), "-m", "agent.terminal"],
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    combined = proc.stdout + proc.stderr
    assert "agent.terminal" in combined or "npm run ASCS" in combined
    assert "python -m agent --tui" not in combined
