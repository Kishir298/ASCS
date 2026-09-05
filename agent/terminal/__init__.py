"""A.S.C.S. terminal interface.

Terminal-native primary UI: entry normalization, curses TUI, slash commands,
pickers, streaming, cancellation. Shared ``EventHub``/``TaskRunner`` live in
``agent.web`` (HTTP serving itself is legacy with a missing-asset fallback).

Canonical implementation: ``agent/terminal/entry.py`` (moved from
``agent/terminal.py``), ``agent/terminal/tui.py`` (moved from
``agent/tui.py``). Old ``agent.tui`` path is preserved via a shim;
``from agent.terminal import …`` (old entry API) is preserved via lazy
re-exports here. New code should import from ``agent.terminal`` submodules.
"""

from __future__ import annotations

_LAZY: dict[str, str] = {
    "ASCII_BANNER": "agent.terminal.entry",
    "normalize_argv": "agent.terminal.entry",
    "main": "agent.terminal.entry",
    "TuiApp": "agent.terminal.tui",
    "run_tui": "agent.terminal.tui",
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
