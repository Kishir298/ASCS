"""Configuration for the coding agent.

Resolution order (highest wins):
    explicit CLI overrides -> environment variables -> defaults.

No Python source edits are required to change configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_MAX_ITERATIONS = 50


@dataclass(frozen=True)
class AgentConfig:
    """Immutable effective configuration for one agent run."""

    workspace: Path = Path.cwd()
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_MODEL
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    mode: str = "AUTO"  # AUTO or SAFE
    verbose: bool = False
    command_timeout: int = 120  # seconds
    max_output_chars: int = 20_000  # per-tool-output truncation limit
    context_budget_chars: int = 70_000  # rolling history budget
    malformed_retry_limit: int = 5
    tools: tuple[str, ...] = field(
        default_factory=lambda: (
            "list_directory",
            "read_file",
            "search_files",
            "write_file",
            "apply_patch",
            "run_command",
            "git_status",
            "git_diff",
        )
    )

    @property
    def is_safe_mode(self) -> bool:
        return self.mode.upper() == "SAFE"


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


def load_config(**overrides) -> AgentConfig:
    """Build an AgentConfig from environment variables and explicit overrides.

    ``overrides`` keys map directly to ``AgentConfig`` field names. Explicit
    values always win over the environment.
    """
    mode = os.environ.get("AGENT_MODE", "AUTO").strip().upper()
    if mode not in ("AUTO", "SAFE"):
        raise ValueError(f"AGENT_MODE must be AUTO or SAFE, got {mode!r}")
    verbose_raw = os.environ.get("AGENT_VERBOSE", "").strip().lower()
    verbose = verbose_raw in ("1", "true", "yes", "on") if verbose_raw else False

    kwargs = {
        "ollama_base_url": _env_str("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        "model": _env_str("OLLAMA_MODEL", DEFAULT_MODEL),
        "max_iterations": _env_int("AGENT_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS),
        "mode": mode,
        "verbose": verbose,
        "command_timeout": _env_int("AGENT_COMMAND_TIMEOUT", 120),
        "max_output_chars": _env_int("AGENT_MAX_OUTPUT_CHARS", 20_000),
        "context_budget_chars": _env_int("AGENT_CONTEXT_BUDGET_CHARS", 70_000),
        "malformed_retry_limit": _env_int("AGENT_MALFORMED_RETRY_LIMIT", 5),
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
    if config.mode.upper() not in ("AUTO", "SAFE"):
        raise ValueError(f"mode must be AUTO or SAFE, got {config.mode!r}")
    if not config.workspace.exists():
        raise ValueError(f"Workspace does not exist: {config.workspace}")
    if not config.workspace.is_dir():
        raise ValueError(f"Workspace is not a directory: {config.workspace}")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if config.command_timeout <= 0:
        raise ValueError("command_timeout must be positive")
    if config.max_output_chars <= 0:
        raise ValueError("max_output_chars must be positive")