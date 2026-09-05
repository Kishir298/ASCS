"""A.S.C.S. models.

Ollama client, provider abstraction, model selection, and model response
handling. Existing Ollama behavior is preserved; locked models remain
``qwen3-coder:30b`` primary + ``qwen2.5-coder:14b`` fallback.

Canonical implementation: ``agent/models/client.py`` (moved from
``agent/ollama.py``), ``agent/models/providers.py``, ``agent/models/
responses.py`` (moved from ``agent/models.py``). Old ``agent.ollama`` /
``agent.providers`` paths are preserved via shims; old
``from agent.models import …`` (response contract) is preserved via lazy
re-exports here.

Re-exports are lazy (PEP 562) to avoid circular imports.
"""

from __future__ import annotations

_LAZY: dict[str, str] = {
    # Response contract (old agent.models API — preserved).
    "ModelReply": "agent.models.responses",
    "Plan": "agent.models.responses",
    "ToolResult": "agent.models.responses",
    "parse_model_reply": "agent.models.responses",
    "tool_result_message": "agent.models.responses",
    "truncate": "agent.models.responses",
    # Ollama client.
    "OllamaClient": "agent.models.client",
    "OllamaError": "agent.models.client",
    "OllamaConnectionError": "agent.models.client",
    "OllamaTimeoutError": "agent.models.client",
    "OllamaHTTPError": "agent.models.client",
    "OllamaModelNotFoundError": "agent.models.client",
    "OllamaResponseError": "agent.models.client",
    "resilient_chat": "agent.models.client",
    # Providers.
    "PROVIDER_NAMES": "agent.models.providers",
    "ProviderInfo": "agent.models.providers",
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
