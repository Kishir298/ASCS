"""Provider connector for local and cloud models.

Supported providers (all optional, Ollama always available):
  - ollama        (local, http://localhost:11434)
  - openai        (https://api.openai.com)
  - anthropic     (https://api.anthropic.com)
  - grok          (xAI, https://api.x.ai)
  - google        (AI Studio, https://generativelanguage.googleapis.com)
  - deepseek      (https://api.deepseek.com)

All providers are Ollama-compatible in the sense that the TUI never
assumes a cloud provider is available — every operation falls back to
Ollama and an empty model list is treated as a valid (empty) state,
never an error that blocks the UI.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

PROVIDER_NAMES = ("ollama", "openai", "anthropic", "grok", "google", "deepseek")
DEFAULT_PROVIDER = "ollama"

PROVIDER_DISPLAY = {
    "ollama": "Ollama (local)",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "grok": "Grok (xAI)",
    "google": "Google AI Studio",
    "deepseek": "DeepSeek",
}

DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "grok": "https://api.x.ai",
    "google": "https://generativelanguage.googleapis.com",
    "deepseek": "https://api.deepseek.com",
}

# Env var that holds the API key for each cloud provider.
API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "grok": "XAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    # ollama has no key
}

CACHE_TTL_S = 300  # 5 minutes


def _cache_path() -> Path:
    return Path.home() / ".risa" / "ascs" / "provider_cache.json"


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    display_name: str
    base_url: str
    api_key_env: str | None


def get_provider_info(name: str, base_url_override: str | None = None) -> ProviderInfo:
    if name not in PROVIDER_NAMES:
        raise ValueError(f"Unknown provider {name!r}; valid: {', '.join(PROVIDER_NAMES)}")
    base = (base_url_override or DEFAULT_BASE_URLS[name]).rstrip("/")
    return ProviderInfo(
        name=name,
        display_name=PROVIDER_DISPLAY[name],
        base_url=base,
        api_key_env=API_KEY_ENVS.get(name),
    )


def get_api_key(provider: str, override: str | None = None) -> str | None:
    if override is not None:
        return override or None
    env_name = API_KEY_ENVS.get(provider)
    if not env_name:
        return None
    val = os.environ.get(env_name, "").strip()
    return val or None


# ---------------------------------------------------------------------------
# Low-level HTTP helper (stdlib only)
# ---------------------------------------------------------------------------

def _get_json(url: str, headers: dict[str, str], timeout: int = 10) -> Any:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # type: ignore[arg-type]
        raw = resp.read()
    return json.loads(raw.decode("utf-8", errors="replace"))


def _safe_list_models_ollama(base_url: str, timeout: int = 10) -> list[str]:
    """List Ollama models via /api/tags. Never raises; returns [] on failure."""
    try:
        data = _get_json(f"{base_url.rstrip('/')}/api/tags", {"Accept": "application/json"}, timeout=timeout)
        models = data.get("models") or []
        out: list[str] = []
        for m in models:
            if isinstance(m, dict) and m.get("name"):
                out.append(str(m["name"]))
        return out
    except Exception:
        return []


def _safe_list_models_openai_compat(base_url: str, api_key: str | None, timeout: int = 10) -> list[str]:
    """OpenAI-compatible /v1/models (openai, grok, deepseek)."""
    try:
        headers: dict[str, str] = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = _get_json(f"{base_url.rstrip('/')}/v1/models", headers, timeout=timeout)
        items = data.get("data") or []
        out: list[str] = []
        for item in items:
            if isinstance(item, dict):
                ident = item.get("id") or item.get("name")
                if ident:
                    out.append(str(ident))
        return out
    except Exception:
        return []


def _safe_list_models_anthropic(base_url: str, api_key: str | None, timeout: int = 10) -> list[str]:
    try:
        if not api_key:
            return []
        headers = {
            "Accept": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        data = _get_json(f"{base_url.rstrip('/')}/v1/models", headers, timeout=timeout)
        items = data.get("data") or []
        out: list[str] = []
        for item in items:
            if isinstance(item, dict):
                ident = item.get("id") or item.get("display_name") or item.get("name")
                if ident:
                    out.append(str(ident))
        return out
    except Exception:
        return []


def _safe_list_models_google(base_url: str, api_key: str | None, timeout: int = 10) -> list[str]:
    try:
        if not api_key:
            return []
        # Google: GET /v1beta/models?key=API_KEY
        # Some deployments use /v1/models?key=
        for path in ("/v1beta/models", "/v1/models"):
            try:
                data = _get_json(f"{base_url.rstrip('/')}{path}?key={api_key}", {"Accept": "application/json"}, timeout=timeout)
                models = data.get("models") or []
                out: list[str] = []
                for m in models:
                    if isinstance(m, dict):
                        name = m.get("name") or m.get("displayName") or ""
                        # Google returns "models/gemini-1.5-pro" -> strip prefix
                        if name.startswith("models/"):
                            name = name[len("models/") :]
                        if name:
                            out.append(str(name))
                if out:
                    return out
            except Exception:
                continue
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public listing API
# ---------------------------------------------------------------------------

def list_models_for_provider(
    provider: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int = 10,
    use_cache: bool = True,
) -> list[str]:
    """Return models for ``provider``. Never raises — returns [] on failure.

    ``base_url`` and ``api_key`` override defaults/env. Results are cached
    for ``CACHE_TTL_S`` unless ``use_cache`` is False.
    """
    provider = provider.strip().lower()
    if provider not in PROVIDER_NAMES:
        return []
    # Try cache first
    cache_key = f"{provider}:{base_url or DEFAULT_BASE_URLS[provider]}"
    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            return cached

    base = (base_url or DEFAULT_BASE_URLS[provider]).rstrip("/")
    key = api_key if api_key is not None else get_api_key(provider)

    if provider == "ollama":
        result = _safe_list_models_ollama(base, timeout=timeout)
    elif provider in ("openai", "grok", "deepseek"):
        result = _safe_list_models_openai_compat(base, key, timeout=timeout)
        # Grok fallback: if generic compat is empty, return curated list
        if not result and provider == "grok" and key:
            # curated fallback so picker is not completely empty when API is
            # temporarily unreachable but key is set
            result = []
    elif provider == "anthropic":
        result = _safe_list_models_anthropic(base, key, timeout=timeout)
    elif provider == "google":
        result = _safe_list_models_google(base, key, timeout=timeout)
    else:
        result = []

    # Google / Anthropic without key -> empty (as required by spec)
    # Don't fabricate models for unauthenticated cloud providers
    if use_cache:
        _save_to_cache(cache_key, result)
    return result


def list_all_providers_with_models(
    *,
    overrides: dict[str, dict[str, str]] | None = None,
    timeout: int = 2,
    use_cache: bool = True,
) -> dict[str, list[str]]:
    """Return {provider: [models]} for every provider in PROVIDER_NAMES.

    ``overrides`` may map provider -> {"base_url": "...", "api_key": "..."}.
    An empty list for a provider is valid and rendered as empty per spec.

    Fetches in parallel via threads so total latency is ~max(single) not sum,
    critical for TUI picker responsiveness (was sequential → 5-10s).
    ``timeout`` is per-provider; 2s keeps picker snappy while still allowing
    local Ollama to respond.
    """
    import concurrent.futures

    # Fast path: if all cached and use_cache, avoid threads entirely
    # Check cache first; only fetch missing/expired
    result: dict[str, list[str]] = {}
    to_fetch: list[str] = []
    for name in PROVIDER_NAMES:
        ov = (overrides or {}).get(name, {})
        cache_key = f"{name}:{ov.get('base_url') or DEFAULT_BASE_URLS[name]}"
        if use_cache:
            cached = _load_from_cache(cache_key)
            if cached is not None:
                result[name] = cached
                continue
        to_fetch.append(name)

    if not to_fetch:
        return result

    # Parallel fetch for remaining
    def _fetch_one(name: str) -> tuple[str, list[str]]:
        ov = (overrides or {}).get(name, {})
        # Use shorter timeout for cloud without key to fail fast; Ollama gets full timeout
        per_timeout = timeout
        # If cloud provider has no API key, skip network entirely -> instant []
        if name != "ollama" and not (ov.get("api_key") or get_api_key(name)):
            # Still respect cache already checked; return [] and cache it to avoid repeat DNS
            models: list[str] = []
            cache_key = f"{name}:{ov.get('base_url') or DEFAULT_BASE_URLS[name]}"
            if use_cache:
                _save_to_cache(cache_key, models)
            return (name, models)
        models = list_models_for_provider(
            name,
            base_url=ov.get("base_url"),
            api_key=ov.get("api_key"),
            timeout=per_timeout,
            use_cache=use_cache,
        )
        return (name, models)

    # Use max 4 workers to avoid flooding; 6 providers is tiny
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
        futs = {exe.submit(_fetch_one, n): n for n in to_fetch}
        for fut in concurrent.futures.as_completed(futs):
            try:
                n, models = fut.result()
                result[n] = models
            except Exception:
                n = futs[fut]
                result[n] = []

    # Ensure all providers present (in case thread failed)
    for name in PROVIDER_NAMES:
        if name not in result:
            result[name] = []
    return result


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_from_cache(key: str) -> list[str] | None:
    try:
        p = _cache_path()
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        entry = data.get(key)
        if not isinstance(entry, dict):
            return None
        ts = entry.get("ts", 0)
        if time.time() - float(ts) > CACHE_TTL_S:
            return None
        models = entry.get("models")
        if isinstance(models, list):
            return [str(m) for m in models]
    except Exception:
        return None
    return None


def _save_to_cache(key: str, models: list[str]) -> None:
    try:
        p = _cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data[key] = {"models": list(models), "ts": time.time()}
        # prune old entries over 20
        if len(data) > 20:
            # keep newest 20
            sorted_items = sorted(data.items(), key=lambda kv: kv[1].get("ts", 0) if isinstance(kv[1], dict) else 0)
            data = dict(sorted_items[-20:])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:
        pass


def clear_cache() -> None:
    try:
        p = _cache_path()
        if p.exists():
            p.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ollama-compat helpers
# ---------------------------------------------------------------------------

def is_ollama_available(base_url: str | None = None, timeout: int = 5) -> bool:
    """True if Ollama responds at ``base_url``. Used as fallback guard."""
    base = (base_url or DEFAULT_BASE_URLS["ollama"]).rstrip("/")
    try:
        _get_json(f"{base}/api/version", {"Accept": "application/json"}, timeout=timeout)
        return True
    except Exception:
        return False


def get_ollama_compat_models(
    provider: str,
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    ollama_base_url: str | None = None,
    timeout: int = 8,
) -> tuple[list[str], bool]:
    """Return (models, is_fallback).

    If cloud ``provider`` returns empty/unreachable, returns Ollama models
    with ``is_fallback=True`` so callers can show a hint.
    """
    if provider == "ollama":
        return (list_models_for_provider("ollama", base_url=ollama_base_url or base_url, timeout=timeout), False)
    cloud = list_models_for_provider(provider, base_url=base_url, api_key=api_key, timeout=timeout)
    if cloud:
        return (cloud, False)
    # fallback to ollama so UI is never blocked
    ol = list_models_for_provider("ollama", base_url=ollama_base_url, timeout=timeout)
    return (ol, True)
