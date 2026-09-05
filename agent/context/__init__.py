"""A.S.C.S. project context.

Project understanding and indexing, independent from planning, execution,
and tools — it supplies project information to those systems.

Canonical implementation: ``agent/context/index.py`` (moved from
``agent/context.py``), ``agent/context/project.py``, ``agent/context/
toolchain.py``. ``from agent.context import …`` preserves the old index
import path via lazy re-exports. Old ``agent.project`` / ``agent.toolchain``
paths are preserved via shims.

Re-exports are lazy (PEP 562) to avoid circular imports between domains.
"""

from __future__ import annotations

_LAZY: dict[str, str] = {
    "ProjectIndex": "agent.context.index",
    "FileRecord": "agent.context.index",
    "Symbol": "agent.context.index",
    "ContextChunk": "agent.context.index",
    "ContextBundle": "agent.context.index",
    "ContextError": "agent.context.index",
    "DEFAULT_STATE_DIR": "agent.context.index",
    "DEFAULT_INDEX_FILE": "agent.context.index",
    "DEFAULT_IGNORED_DIRS": "agent.context.index",
    "DEFAULT_CHUNK_TOKENS": "agent.context.index",
    "DEFAULT_CHUNK_TOKENS_30B": "agent.context.index",
    "DEFAULT_CHUNK_TOKENS_14B": "agent.context.index",
    "CHUNK_TOKENS_BY_MODEL": "agent.context.index",
    "create_project_index": "agent.context.index",
    "git_status": "agent.context.index",
    "ProjectManifest": "agent.context.project",
    "ProjectScanner": "agent.context.project",
    "ProjectStore": "agent.context.project",
    "scan": "agent.context.project",
    "project_prompt_text": "agent.context.project",
    "Toolchain": "agent.context.toolchain",
    "detect_toolchain": "agent.context.toolchain",
    "toolchain_to_text": "agent.context.toolchain",
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
