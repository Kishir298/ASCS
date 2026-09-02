"""Configuration for the coding agent.

Resolution order (highest wins):
    explicit CLI overrides -> environment variables -> TUI persisted state -> defaults.

No Python source edits are required to change configuration.

Modes
=====
The agent supports three primary user modes:

    PLAN  - inspect the workspace and produce a structured implementation
            plan. No workspace files are modified.
    BUILD - execute an approved implementation: modify files, run tests, fix
            failures, verify results.
    AUTO  - receive a high-level request, plan it, then plan+execute+test+
            verify fully autonomously. This is the primary autonomous mode.

``SAFE`` is retained as a backward-compatible *approval overlay* on top of the
primary modes: when enabled, modification/command tools prompt the operator
for approval before running.

Provider & Intelligence
=======================
``provider`` selects the model host (``ollama`` always available, plus
``openai``, ``anthropic``, ``grok``, ``google``, ``deepseek``). ``intelligence``
is a tier that controls the generation budget (``num_ctx``, ``num_predict``,
``context_budget_chars`` and the retrieval depth). ``theme`` controls the TUI
appearance (``auto``/``light``/``dark``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-coder:30b"  # primary 30B coder, fallback qwen2.5-coder:14b via --model
DEFAULT_REQUEST_TIMEOUT = 600  # seconds; single model request budget
DEFAULT_KEEP_ALIVE = "30m"  # Ollama model persistence between requests
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_UI_HOST = "127.0.0.1"
DEFAULT_UI_PORT = 8787

# Qwen3 generation knobs (sent to the native Ollama endpoint as options).
# Bumped for qwen3-coder:30b (262k ctx) — chunking splits 300k tasks into feasible windows.
DEFAULT_NUM_CTX = 65536  # context window size (30B max feasible on 32GB)
DEFAULT_NUM_PREDICT = 16384  # max tokens generated per request

# Ollama client retry / backoff policy for transient failures.
DEFAULT_MAX_RETRIES = 2  # retries after the initial attempt (=3 total attempts)
DEFAULT_BACKOFF_S = 2.0

# The three user-facing modes.
MODES = ("PLAN", "BUILD", "AUTO")

# Provider and intelligence.
PROVIDER_NAMES = ("ollama", "openai", "anthropic", "grok", "google", "deepseek")
DEFAULT_PROVIDER = "ollama"
INTELLIGENCE_LEVELS = ("low", "medium", "high", "xhigh", "default")
DEFAULT_INTELLIGENCE = "default"
DEFAULT_THEME = "auto"
THEMES = ("auto", "light", "dark")

# Intelligence tier -> (num_ctx, num_predict, context_budget_chars, retrieve_level)
# Max-chunking for 300k-token tasks: larger windows + higher budgets; default===high
# (fallback qwen2.5-coder:14b still works via low/medium tiers on 16GB)
INTELLIGENCE_MAP: dict[str, tuple[int, int, int, int]] = {
    "low": (8192, 2048, 30000, 1),
    "medium": (16384, 4096, 50000, 2),
    "high": (65536, 16384, 90000, 3),
    "xhigh": (131072, 32768, 140000, 4),
    "default": (65536, 16384, 90000, 3),
}

# Tools that modify the workspace / run arbitrary commands. In PLAN mode these
# are removed from the enabled set; in SAFE mode they are gated behind
# operator approval. This is the single source of truth for "modifying" tools.
MODIFY_TOOLS = frozenset(
    {
        "write_file",
        "apply_patch",
        "delete_file",
        "move_file",
        "copy_file",
        "run_command",
    }
)

# Backward-compatible alias: PLAN mode disables the modifying tools.
PLAN_MODE_BLOCKED = MODIFY_TOOLS

# Tools the agent may use without operator approval even in SAFE mode.
READONLY_TOOLS = {
    "list_directory",
    "read_file",
    "search_files",
    "inspect_environment",
    "git_status",
    "git_diff",
    "set_plan",
}


@dataclass(frozen=True)
class AgentConfig:
    """Immutable effective configuration for one agent run."""

    workspace: Path = Path.cwd()
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER  # ollama | openai | anthropic | grok | google | deepseek
    intelligence: str = DEFAULT_INTELLIGENCE  # low | medium | high | xhigh | default
    theme: str = DEFAULT_THEME  # auto | light | dark
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    mode: str = "AUTO"  # PLAN, BUILD or AUTO (SAFE accepted as overlay)
    approval: bool = False  # True -> prompt before modifications/commands
    verbose: bool = False
    command_timeout: int = 120  # seconds
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT  # seconds; single model request budget
    keep_alive: str | None = DEFAULT_KEEP_ALIVE  # Ollama keep_alive e.g. "30m"
    prewarm: bool = True  # warm the model before the first real step
    num_ctx: int = DEFAULT_NUM_CTX  # Qwen3 context window size
    num_predict: int = DEFAULT_NUM_PREDICT  # max tokens generated per request
    max_retries: int = DEFAULT_MAX_RETRIES  # retries past the initial attempt
    backoff_s: float = DEFAULT_BACKOFF_S  # base backoff seconds between retries
    max_output_chars: int = 20_000  # per-tool-output truncation limit
    context_budget_chars: int = 70_000  # rolling history budget
    max_verify_retries: int = 2  # verification failure retries per task
    malformed_retry_limit: int = 5
    experience_enabled: bool = False  # persist/retrieve experience across runs
    experience_path: str | None = None  # custom experience store path (default: ~/.risa/ascs)
    ui_host: str = DEFAULT_UI_HOST
    ui_port: int = DEFAULT_UI_PORT
    tools: tuple[str, ...] = field(
        default_factory=lambda: (
            "list_directory",
            "read_file",
            "search_files",
            "write_file",
            "apply_patch",
            "delete_file",
            "move_file",
            "copy_file",
            "run_command",
            "inspect_environment",
            "git_status",
            "git_diff",
            "set_plan",
        )
    )

    @property
    def is_safe_mode(self) -> bool:
        """True when modifications/commands require operator approval."""
        return self.approval or self.mode.upper() == "SAFE"

    @property
    def primary_mode(self) -> str:
        """The user-facing mode, mapping legacy SAFE onto AUTO."""
        if self.is_plan_mode:
            return "PLAN"
        if self.is_build_mode:
            return "BUILD"
        return "AUTO"

    @property
    def is_plan_mode(self) -> bool:
        return self.mode.upper() == "PLAN"

    @property
    def is_build_mode(self) -> bool:
        return self.mode.upper() == "BUILD"

    @property
    def is_auto_mode(self) -> bool:
        return self.mode.upper() in ("AUTO", "SAFE")

    @property
    def effective_tools(self) -> tuple[str, ...]:
        """The tools actually available to the loop for this run."""
        names = self.tools
        if self.is_plan_mode:
            names = tuple(n for n in names if n not in PLAN_MODE_BLOCKED)
        return names

    @property
    def retrieve_level(self) -> int:
        """Context retrieval depth derived from intelligence tier."""
        return INTELLIGENCE_MAP.get(self.intelligence, INTELLIGENCE_MAP[DEFAULT_INTELLIGENCE])[3]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {name} must be positive, got {value}")
    return value


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    lowered = raw.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Environment variable {name} must be a boolean, got {raw!r}")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"Environment variable {name} must be non-negative, got {value}")
    return value


# ---------------------------------------------------------------------------
# TUI persistence helpers
# ---------------------------------------------------------------------------

def tui_state_path() -> Path:
    """Path to the persisted TUI state (provider/model/intelligence/theme).

    Overridable via ``AGENT_TUI_STATE_PATH`` for testing.
    """
    custom = os.environ.get("AGENT_TUI_STATE_PATH", "").strip()
    if custom:
        return Path(custom).expanduser()
    return Path.home() / ".risa" / "ascs" / "tui_state.json"


def load_tui_state(path: Path | None = None) -> dict:
    """Load persisted TUI state; never raises, returns {} on failure."""
    p = path or tui_state_path()
    try:
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_tui_state(state: dict, path: Path | None = None) -> None:
    """Persist TUI state atomically. Creates parent dirs, 0o600 best-effort."""
    p = path or tui_state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        # Merge with existing to avoid clobbering concurrent provider configs
        try:
            existing = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}
        merged = {**existing, **state}
        tmp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        try:
            tmp.chmod(0o600)
        except Exception:
            pass
        os.replace(tmp, p)
        try:
            p.chmod(0o600)
        except Exception:
            pass
    except Exception:
        pass


def intelligence_values(level: str) -> tuple[int, int, int, int]:
    """Return (num_ctx, num_predict, context_budget, retrieve_level) for level."""
    lvl = (level or DEFAULT_INTELLIGENCE).strip().lower()
    return INTELLIGENCE_MAP.get(lvl, INTELLIGENCE_MAP[DEFAULT_INTELLIGENCE])


def _apply_intelligence_defaults(kwargs: dict, level: str) -> None:
    """Fill num_ctx / num_predict / context_budget from intelligence tier
    only when caller has not already set them (CLI/env wins)."""
    lvl = (level or DEFAULT_INTELLIGENCE).strip().lower()
    if lvl not in INTELLIGENCE_MAP:
        lvl = DEFAULT_INTELLIGENCE
    n_ctx, n_pred, c_budget, _ = INTELLIGENCE_MAP[lvl]
    # Only fill if not already provided via env/CLI (kwargs already contains env defaults,
    # but we want intelligence to override the bare defaults only when env didn't set them).
    # To detect env override we check whether env var was set.
    if "AGENT_NUM_CTX" not in os.environ and "num_ctx" not in kwargs:
        kwargs["num_ctx"] = n_ctx
    elif lvl != DEFAULT_INTELLIGENCE and "AGENT_NUM_CTX" not in os.environ and kwargs.get("num_ctx") == DEFAULT_NUM_CTX:
        # If env not set but kwargs has default value, allow intelligence to override
        # unless caller explicitly passed num_ctx via overrides (handled after).
        pass  # will be handled below with explicit check
    # Simpler: if AGENT_NUM_CTX env not set, apply intelligence value unconditionally
    # unless overrides already contains num_ctx from CLI.
    # We do this by checking if num_ctx was provided explicitly in overrides later,
    # so for now just set if env not present.
    if "AGENT_NUM_CTX" not in os.environ:
        # will be overridden by kwargs.update(overrides) if CLI set it, so safe to set now
        kwargs["num_ctx"] = n_ctx
    if "AGENT_NUM_PREDICT" not in os.environ:
        kwargs["num_predict"] = n_pred
    if "AGENT_CONTEXT_BUDGET_CHARS" not in os.environ:
        kwargs["context_budget_chars"] = c_budget


def load_config(**overrides) -> AgentConfig:
    """Build an AgentConfig from environment variables and explicit overrides.

    ``overrides`` keys map directly to ``AgentConfig`` field names. Explicit
    values always win over the environment, which wins over the persisted TUI
    state, which wins over defaults.
    """
    # Load persisted TUI state for fallback defaults (provider/model/intelligence/theme).
    persisted = load_tui_state()

    mode_raw = os.environ.get("AGENT_MODE", persisted.get("mode", "AUTO")).strip().upper()
    # Allow persisted mode to be overridden by env; default AUTO.
    if mode_raw not in (*MODES, "SAFE"):
        raise ValueError(
            f"AGENT_MODE must be one of {', '.join(MODES)} or SAFE, got {mode_raw!r}"
        )

    # Provider / intelligence / theme: env > persisted > default
    provider_raw = os.environ.get("AGENT_PROVIDER", persisted.get("provider", DEFAULT_PROVIDER)).strip().lower()
    if provider_raw not in PROVIDER_NAMES:
        # Don't crash on old persisted values; fall back
        provider_raw = DEFAULT_PROVIDER
    intelligence_raw = os.environ.get("AGENT_INTELLIGENCE", persisted.get("intelligence", DEFAULT_INTELLIGENCE)).strip().lower()
    if intelligence_raw not in INTELLIGENCE_LEVELS:
        intelligence_raw = DEFAULT_INTELLIGENCE
    theme_raw = os.environ.get("AGENT_THEME", persisted.get("theme", DEFAULT_THEME)).strip().lower()
    if theme_raw not in THEMES:
        theme_raw = DEFAULT_THEME

    # SAFE is a legacy approval overlay; approval is derived from it unless
    # explicitly overridden.
    approval_raw = _env_bool("AGENT_APPROVAL", mode_raw == "SAFE" or False)

    # Intelligence tier influences generation knobs. We resolve it before
    # filling num_ctx etc so env-provided num_ctx still wins.
    intel_n_ctx, intel_n_pred, intel_c_budget, _ = intelligence_values(intelligence_raw)

    # Determine base kwargs from env (with intelligence-aware defaults where env not set)
    # For num_ctx etc we want: if env sets them, use env; else use intelligence values.
    if "AGENT_NUM_CTX" in os.environ:
        num_ctx_val = _env_int("AGENT_NUM_CTX", DEFAULT_NUM_CTX)
    else:
        # Check persisted num_ctx if present and intelligence not explicitly changed via env
        if "num_ctx" in persisted and intelligence_raw == persisted.get("intelligence", DEFAULT_INTELLIGENCE):
            try:
                num_ctx_val = int(persisted["num_ctx"])
            except Exception:
                num_ctx_val = intel_n_ctx
        else:
            num_ctx_val = intel_n_ctx

    if "AGENT_NUM_PREDICT" in os.environ:
        num_pred_val = _env_int("AGENT_NUM_PREDICT", DEFAULT_NUM_PREDICT)
    else:
        if "num_predict" in persisted and intelligence_raw == persisted.get("intelligence", DEFAULT_INTELLIGENCE):
            try:
                num_pred_val = int(persisted["num_predict"])
            except Exception:
                num_pred_val = intel_n_pred
        else:
            num_pred_val = intel_n_pred

    if "AGENT_CONTEXT_BUDGET_CHARS" in os.environ:
        c_budget_val = _env_int("AGENT_CONTEXT_BUDGET_CHARS", 70_000)
    else:
        if "context_budget_chars" in persisted and intelligence_raw == persisted.get("intelligence", DEFAULT_INTELLIGENCE):
            try:
                c_budget_val = int(persisted["context_budget_chars"])
            except Exception:
                c_budget_val = intel_c_budget
        else:
            c_budget_val = intel_c_budget

    kwargs = {
        "ollama_base_url": _env_str("OLLAMA_BASE_URL", persisted.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)),
        "model": _env_str("OLLAMA_MODEL", persisted.get("model", DEFAULT_MODEL)),
        "provider": provider_raw,
        "intelligence": intelligence_raw,
        "theme": theme_raw,
        "max_iterations": _env_int("AGENT_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS),
        "mode": mode_raw,
        "approval": approval_raw,
        "verbose": _env_bool("AGENT_VERBOSE", False),
        "command_timeout": _env_int("AGENT_COMMAND_TIMEOUT", 120),
        "request_timeout": _env_int("AGENT_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT),
        "keep_alive": _env_str("AGENT_KEEP_ALIVE", DEFAULT_KEEP_ALIVE) or None,
        "prewarm": _env_bool("AGENT_PREWARM", True),
        "num_ctx": num_ctx_val,
        "num_predict": num_pred_val,
        "max_retries": _env_int("AGENT_MAX_RETRIES", DEFAULT_MAX_RETRIES),
        "backoff_s": _env_float("AGENT_BACKOFF_S", DEFAULT_BACKOFF_S),
        "max_output_chars": _env_int("AGENT_MAX_OUTPUT_CHARS", 20_000),
        "context_budget_chars": c_budget_val,
        "malformed_retry_limit": _env_int("AGENT_MALFORMED_RETRY_LIMIT", 5),
        "max_verify_retries": _env_int("AGENT_MAX_VERIFY_RETRIES", 2),
        "experience_enabled": _env_bool("AGENT_EXPERIENCE_ENABLED", True),
        "experience_path": _env_str("AGENT_EXPERIENCE_PATH", "") or None,
        "ui_host": _env_str("AGENT_UI_HOST", DEFAULT_UI_HOST),
        "ui_port": _env_int("AGENT_UI_PORT", DEFAULT_UI_PORT),
    }
    kwargs.update(overrides)

    # If intelligence was overridden explicitly, recompute generation knobs
    # unless the caller also explicitly set those knobs or env did.
    if "intelligence" in overrides:
        new_intel = overrides["intelligence"]
        if isinstance(new_intel, str) and new_intel.strip().lower() in INTELLIGENCE_MAP:
            new_intel = new_intel.strip().lower()
            n_ctx, n_pred, c_budget, _ = INTELLIGENCE_MAP[new_intel]
            if "num_ctx" not in overrides and "AGENT_NUM_CTX" not in os.environ:
                kwargs["num_ctx"] = n_ctx
            if "num_predict" not in overrides and "AGENT_NUM_PREDICT" not in os.environ:
                kwargs["num_predict"] = n_pred
            if "context_budget_chars" not in overrides and "AGENT_CONTEXT_BUDGET_CHARS" not in os.environ:
                kwargs["context_budget_chars"] = c_budget

    if "workspace" in kwargs:
        kwargs["workspace"] = Path(kwargs["workspace"]).expanduser()
    else:
        kwargs["workspace"] = Path.cwd()

    config = AgentConfig(**kwargs)
    _validate(config)
    return config


def _validate(config: AgentConfig) -> None:
    if config.mode.upper() not in (*MODES, "SAFE"):
        raise ValueError(
            f"mode must be one of {', '.join(MODES)} or SAFE, got {config.mode!r}"
        )
    if config.provider not in PROVIDER_NAMES:
        raise ValueError(
            f"provider must be one of {', '.join(PROVIDER_NAMES)}, got {config.provider!r}"
        )
    if config.intelligence not in INTELLIGENCE_LEVELS:
        raise ValueError(
            f"intelligence must be one of {', '.join(INTELLIGENCE_LEVELS)}, got {config.intelligence!r}"
        )
    if config.theme not in THEMES:
        raise ValueError(
            f"theme must be one of {', '.join(THEMES)}, got {config.theme!r}"
        )
    if not config.workspace.exists():
        raise ValueError(f"Workspace does not exist: {config.workspace}")
    if not config.workspace.is_dir():
        raise ValueError(f"Workspace is not a directory: {config.workspace}")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if config.command_timeout <= 0:
        raise ValueError("command_timeout must be positive")
    if config.request_timeout <= 0:
        raise ValueError("request_timeout must be positive")
    if config.max_output_chars <= 0:
        raise ValueError("max_output_chars must be positive")
    if config.num_ctx <= 0:
        raise ValueError("num_ctx must be positive")
    if config.num_predict <= 0:
        raise ValueError("num_predict must be positive")
    if config.max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if config.backoff_s < 0:
        raise ValueError("backoff_s must be non-negative")
    from .tools import TOOL_SPECS

    unknown = set(config.tools) - set(TOOL_SPECS)
    if unknown:
        raise ValueError(f"config.tools contains unknown tools: {sorted(unknown)}")
