"""A.S.C.S. tools.

Actual tool implementations and dispatch: filesystem, search, git, shell,
environment. Categories are logical (see ``agent/tools/core.py``); Phase 3
owns PowerShell hardening. No fake implementations.

Canonical implementation: ``agent/tools/core.py`` (moved from
``agent/tools.py`` in Phase 0). ``from agent.tools import …`` preserves the
old import path via lazy re-exports.
"""

from __future__ import annotations

_LAZY: dict[str, str] = {
    "TOOL_SPECS": "agent.tools.core",
    "ToolSpec": "agent.tools.core",
    "ToolValidationError": "agent.tools.core",
    "get_tool_spec": "agent.tools.core",
    "execute_tool": "agent.tools.core",
    "tool_schema_text": "agent.tools.core",
    "validate_tool_call": "agent.tools.core",
    "TRUNCATION_MARKER": "agent.tools.core",
    "truncate_env": "agent.tools.core",
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
