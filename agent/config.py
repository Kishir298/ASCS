"""Configuration for the coding agent.

Resolution order (highest wins):
    explicit CLI overrides -> environment variables -> defaults.

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
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:14b"
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_UI_HOST = "127.0.0.1"
DEFAULT_UI_PORT = 8787

# Qwen3 generation knobs (sent to the native Ollama endpoint as options).
DEFAULT_NUM_CTX = 32768  # context window size
DEFAULT_NUM_PREDICT = 8192  # max tokens generated per request

# Ollama client retry / backoff policy for transient failures.
DEFAULT_MAX_RETRIES = 2  # retries after the initial attempt (=3 total attempts)
DEFAULT_BACKOFF_S = 2.0

# The three user-facing modes.
MODES = ("PLAN", "BUILD", "AUTO")

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
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    mode: str = "AUTO"  # PLAN, BUILD or AUTO (SAFE accepted as overlay)
    approval: bool = False  # True -> prompt before modifications/commands
    verbose: bool = False
    command_timeout: int = 120  # seconds
    request_timeout: int = 600  # seconds; single model request budget
    keep_alive: str | None = None  # Ollama keep_alive e.g. "30m"
    prewarm: bool = True  # warm the model before the first real step
    num_ctx: int = DEFAULT_NUM_CTX  # Qwen3 context window size
    num_predict: int = DEFAULT_NUM_PREDICT  # max tokens generated per request
    max_retries: int = DEFAULT_MAX_RETRIES  # retries past the initial attempt
    backoff_s: float = DEFAULT_BACKOFF_S  # base backoff seconds between retries
    max_output_chars: int = 20_000  # per-tool-output truncation limit
    context_budget_chars: int = 70_000  # rolling history budget
    max_verify_retries: int = 2  # verification failure retries per task
    malformed_retry_limit: int = 5
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


def load_config(**overrides) -> AgentConfig:
    """Build an AgentConfig from environment variables and explicit overrides.

    ``overrides`` keys map directly to ``AgentConfig`` field names. Explicit
    values always win over the environment.
    """
    mode_raw = os.environ.get("AGENT_MODE", "AUTO").strip().upper()
    if mode_raw not in (*MODES, "SAFE"):
        raise ValueError(
            f"AGENT_MODE must be one of {', '.join(MODES)} or SAFE, got {mode_raw!r}"
        )

    # SAFE is a legacy approval overlay; approval is derived from it unless
    # explicitly overridden.
    approval_raw = _env_bool("AGENT_APPROVAL", mode_raw == "SAFE" or False)
    kwargs = {
        "ollama_base_url": _env_str("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        "model": _env_str("OLLAMA_MODEL", DEFAULT_MODEL),
        "max_iterations": _env_int("AGENT_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS),
        "mode": mode_raw,
        "approval": approval_raw,
        "verbose": _env_bool("AGENT_VERBOSE", False),
        "command_timeout": _env_int("AGENT_COMMAND_TIMEOUT", 120),
        "request_timeout": _env_int("AGENT_REQUEST_TIMEOUT", 600),
        "keep_alive": _env_str("AGENT_KEEP_ALIVE", "") or None,
        "prewarm": _env_bool("AGENT_PREWARM", True),
        "num_ctx": _env_int("AGENT_NUM_CTX", DEFAULT_NUM_CTX),
        "num_predict": _env_int("AGENT_NUM_PREDICT", DEFAULT_NUM_PREDICT),
        "max_retries": _env_int("AGENT_MAX_RETRIES", DEFAULT_MAX_RETRIES),
        "backoff_s": _env_float("AGENT_BACKOFF_S", DEFAULT_BACKOFF_S),
        "max_output_chars": _env_int("AGENT_MAX_OUTPUT_CHARS", 20_000),
        "context_budget_chars": _env_int("AGENT_CONTEXT_BUDGET_CHARS", 70_000),
        "malformed_retry_limit": _env_int("AGENT_MALFORMED_RETRY_LIMIT", 5),
        "max_verify_retries": _env_int("AGENT_MAX_VERIFY_RETRIES", 2),
        "ui_host": _env_str("AGENT_UI_HOST", DEFAULT_UI_HOST),
        "ui_port": _env_int("AGENT_UI_PORT", DEFAULT_UI_PORT),
    }
    kwargs.update(overrides)

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