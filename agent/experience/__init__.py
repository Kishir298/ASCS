"""A.S.C.S. experience memory.

Cross-project learning: structured outcomes, relevance search, contradiction
penalties, bounded prompt formatting. Storage, retrieval, formatting, and
persistence behavior are preserved from ``agent/experience.py``.

Canonical implementation: ``agent/experience/store.py``. Old
``from agent.experience import …`` path is preserved via lazy re-exports.
Phase 5 owns the full learning architecture; Phase 0 only provides the home.
"""

from __future__ import annotations

_LAZY: dict[str, str] = {
    "Experience": "agent.experience.store",
    "ExperienceStore": "agent.experience.store",
    "format_for_prompt": "agent.experience.store",
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
