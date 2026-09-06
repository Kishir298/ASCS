"""Ollama server process management.

A.S.C.S. talks to Ollama as an HTTP client and never spawns ``ollama serve``
itself. When the CLI closes, the server is stopped best-effort so a ~19 GB
model does not linger in memory:

1. ``ollama stop <model>`` unloads the model from VRAM;
2. the server process is then terminated
   (Windows ``taskkill /F /IM ollama.exe``, POSIX ``pkill -x ollama``).

Everything here is best-effort and never raises: stopping must never fail
(or delay) CLI shutdown. Set ``AGENT_STOP_OLLAMA_ON_EXIT=false`` to opt out.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable


def stop_on_exit_enabled() -> bool:
    """True unless explicitly opted out via ``AGENT_STOP_OLLAMA_ON_EXIT``."""
    raw = os.environ.get("AGENT_STOP_OLLAMA_ON_EXIT", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def stop_ollama_server(
    model: str,
    *,
    run: Callable[..., object] | None = None,
    log: Callable[[str], None] | None = None,
) -> bool:
    """Unload ``model`` and terminate the Ollama server process.

    Returns True when both steps were attempted without error; False
    otherwise (including when already stopped). Never raises.
    """
    runner = run or subprocess.run
    tell = log or (lambda _msg: None)
    try:
        tell(f"Unloading Ollama model {model!r}…")
        runner(
            ["ollama", "stop", model],
            capture_output=True,
            timeout=60,
        )
        tell("Stopping Ollama server…")
        if os.name == "nt":
            runner(
                ["taskkill", "/F", "/IM", "ollama.exe"],
                capture_output=True,
                timeout=60,
            )
        else:
            runner(
                ["pkill", "-x", "ollama"],
                capture_output=True,
                timeout=60,
            )
    except Exception as exc:  # noqa: BLE001 - shutdown must never fail
        tell(f"Ollama stop skipped: {exc}")
        return False
    return True


def maybe_stop_ollama_on_exit(
    config,
    *,
    log: Callable[[str], None] | None = None,
) -> None:
    """Stop the Ollama server if ``config.stop_ollama_on_exit`` allows it.

    Reads the live ``AGENT_STOP_OLLAMA_ON_EXIT`` env override on every call
    so tests and operators can flip it without rebuilding config. Never
    raises.
    """
    try:
        allowed = bool(getattr(config, "stop_ollama_on_exit", True))
        if not (allowed and stop_on_exit_enabled()):
            return
        stop_ollama_server(
            getattr(config, "model", ""),
            log=log,
        )
    except Exception:  # noqa: BLE001 - shutdown must never fail
        pass
