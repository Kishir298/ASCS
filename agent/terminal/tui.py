"""Interactive TUI for A.S.C.S. — real terminal shell.

OpenCode-inspired terminal UI with:
  - TAB: PLAN -> BUILD -> AUTO
  - Live slash-command autocomplete
  - /models: provider-aware model picker
  - /connect: provider connector
  - /intel: intelligence picker
  - Real AgentLoop / TaskRunner execution
  - Real streaming event handling
  - Persistent TUI state
  - Responsive terminal layouts
  - No fake preview, demo queue, or simulated execution

Zero extra runtime dependencies beyond curses
(windows-curses on Windows).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.config import (
    AgentConfig,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    INTELLIGENCE_LEVELS,
    PROVIDER_NAMES,
    intelligence_values,
    load_config,
    load_tui_state,
    save_tui_state,
)

# ---------------------------------------------------------------------------
# Compatibility constants
# ---------------------------------------------------------------------------

HELLO_TEXT = "Hello, hello, hello, hello, hello, hello, hello, hello, hello"
HELLO_LEN = len(HELLO_TEXT)

MIN_CHATBOX_INNER_W = HELLO_LEN
MIN_CHATBOX_W = MIN_CHATBOX_INNER_W + 2
MIN_CHATBOX_H = 5
MIN_TERM_W = 40
MIN_TERM_H = 10

MODE_ORDER = ("PLAN", "BUILD", "AUTO")

MODE_COLORS = {
    "PLAN": "orange",
    "BUILD": "blue",
    "AUTO": "red",
}

MODE_COLOR_IDX = {
    "PLAN": 208,
    "BUILD": 27,
    "AUTO": 196,
}

PINK_BG_IDX = 213
PINK_FG_IDX = 16

INTEL_CHOICES = ("low", "medium", "high", "xhigh", "default")
INTEL_DISPLAY_ORDER = ("default", "low", "medium", "high", "xhigh")

try:
    import curses  # type: ignore
    import curses.textpad  # noqa: F401

    HAS_CURSES = True
except Exception:  # pragma: no cover
    curses = None  # type: ignore[assignment]
    HAS_CURSES = False


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def next_mode(current: str) -> str:
    """Cycle PLAN -> BUILD -> AUTO -> PLAN."""
    cur = (current or "").strip().upper()

    if cur not in MODE_ORDER:
        return "PLAN"

    idx = MODE_ORDER.index(cur)
    return MODE_ORDER[(idx + 1) % len(MODE_ORDER)]


def format_model_footer(model: str, intelligence: str) -> str:
    """Return model(intelligence)."""
    return f"{model}({intelligence})"


def get_layout_tier(term_h: int, term_w: int) -> str:
    """Return a responsive terminal layout tier."""
    if term_w < MIN_TERM_W or term_h < MIN_TERM_H:
        return "extremely_small"

    if term_h < 12 or term_w < 50:
        return "minimised"

    if term_w < 70:
        return "compact"

    if term_w < 100:
        return "normal"

    if term_w < 140:
        return "large"

    return "wide"


def calc_chatbox_geometry(term_h: int, term_w: int) -> dict[str, int]:
    """Calculate compatibility conversation geometry."""
    tier = get_layout_tier(term_h, term_w)

    is_min = int(tier in ("minimised", "extremely_small"))
    too_small = int(tier == "extremely_small")

    if is_min:
        return {
            "chat_h": 0,
            "chat_w": 0,
            "chat_y": 0,
            "chat_x": 0,
            "is_minimised": 1,
            "inner_w": 0,
            "inner_h": 0,
            "tier": tier,
            "too_small": too_small,
        }

    if tier == "compact":
        side_margin = 0
        avail_w = term_w

    elif tier in ("large", "wide"):
        side_margin = 2
        avail_w = term_w - side_margin * 2
        max_chat_w = 110 if tier == "large" else 120
        avail_w = min(avail_w, max_chat_w)

    else:
        side_margin = 1
        avail_w = term_w - side_margin * 2

    avail_h = term_h - 5

    chat_h = max(
        MIN_CHATBOX_H,
        min(avail_h, term_h - 5),
    )

    if term_h >= 30:
        chat_h = min(
            chat_h,
            max(MIN_CHATBOX_H, term_h // 3 + 2),
        )

    chat_w = max(MIN_CHATBOX_W, avail_w)

    chat_w = min(chat_w, max(1, term_w))

    chat_x = max(0, (term_w - chat_w) // 2)
    chat_y = 1

    inner_w = max(1, chat_w - 2)
    inner_h = max(1, chat_h - 2)

    return {
        "chat_h": chat_h,
        "chat_w": chat_w,
        "chat_y": chat_y,
        "chat_x": chat_x,
        "is_minimised": 0,
        "inner_w": inner_w,
        "inner_h": inner_h,
        "tier": tier,
        "too_small": 0,
    }


def is_minimised(term_h: int, term_w: int) -> bool:
    return bool(calc_chatbox_geometry(term_h, term_w)["is_minimised"])


def is_too_small(term_h: int, term_w: int) -> bool:
    return get_layout_tier(term_h, term_w) == "extremely_small"


def detect_theme(config_theme: str) -> str:
    """Resolve auto theme using terminal/environment hints."""
    theme = (config_theme or "auto").lower()

    if theme in ("light", "dark"):
        return theme

    colorfgbg = os.environ.get("COLORFGBG", "")

    if colorfgbg:
        parts = colorfgbg.replace(":", ";").split(";")

        if parts:
            try:
                bg = int(parts[-1])

                if 7 <= bg <= 15:
                    return "light"

                if bg <= 6:
                    return "dark"

            except ValueError:
                pass

    term = os.environ.get("TERM", "").lower()

    if "light" in term:
        return "light"

    term_program = os.environ.get("TERM_PROGRAM", "").lower()

    if "vscode" in term_program:
        return "dark"

    if "apple_terminal" in term_program:
        return "dark"

    return "dark"


def theme_colors(theme: str) -> dict[str, Any]:
    """Return terminal palette information."""
    resolved = detect_theme(theme)

    if resolved == "light":
        return {
            "theme": "light",
            "bg": "white",
            "bg_idx": 15,
            "fg": "black",
            "fg_idx": 0,
            "chatbox_bg": "grey_light",
            "chatbox_bg_idx": 250,
            "input_fg": "black",
            "input_fg_idx": 0,
            "border_fg": "black",
            "border_fg_idx": 0,
        }

    return {
        "theme": "dark",
        "bg": "black",
        "bg_idx": 16,
        "fg": "white",
        "fg_idx": 15,
        "chatbox_bg": "grey_dark",
        "chatbox_bg_idx": 235,
        "input_fg": "white",
        "input_fg_idx": 15,
        "border_fg": "white",
        "border_fg_idx": 15,
    }


def validate_intel(level: str) -> str:
    """Validate intelligence level."""
    lvl = (level or "").strip().lower()

    if lvl not in INTEL_CHOICES:
        raise ValueError(
            "intelligence must be one of "
            + ", ".join(INTEL_CHOICES)
            + f", got {level!r}"
        )

    return lvl


@dataclass
class PickerItem:
    kind: str
    provider: str
    label: str
    is_provider_header: bool = False


def build_picker_items(
    provider_models: dict[str, list[str]],
) -> list[PickerItem]:
    """Flatten provider/model mapping into picker rows."""
    items: list[PickerItem] = []

    for provider in PROVIDER_NAMES:
        items.append(
            PickerItem(
                kind="provider",
                provider=provider,
                label=provider,
                is_provider_header=True,
            )
        )

        for model in provider_models.get(provider, []) or []:
            items.append(
                PickerItem(
                    kind="model",
                    provider=provider,
                    label=model,
                    is_provider_header=False,
                )
            )

    return items


def build_scoped_picker_items(
    provider_models: dict[str, list[str]],
    active_provider: str,
) -> list[PickerItem]:
    """Build picker rows for only the active provider."""
    provider = (
        (active_provider or "").strip().lower()
        or DEFAULT_PROVIDER
    )

    items = [
        PickerItem(
            kind="provider",
            provider=provider,
            label=provider,
            is_provider_header=True,
        )
    ]

    for model in provider_models.get(provider, []) or []:
        items.append(
            PickerItem(
                kind="model",
                provider=provider,
                label=model,
                is_provider_header=False,
            )
        )

    return items


COMFORTABLE_TIER = "normal"


def is_comfortable_layout(tier: str) -> bool:
    return tier == COMFORTABLE_TIER


def chatbox_bottom_layout(
    mode: str,
    model: str,
    intelligence: str,
    chat_w: int,
    inner_w: int,
    tier: str = "normal",
) -> tuple[str, str, int]:
    """Calculate compatibility footer geometry."""
    footer = format_model_footer(model, intelligence)
    mode_str = f" {mode} "

    if tier == "compact":
        max_footer = inner_w - 8

        if len(footer) > max_footer:
            footer = (
                footer[: max(0, max_footer - 1)]
                + "…"
            )

    if len(footer) + 4 < inner_w:
        footer_x = chat_w - len(footer) - 3

    else:
        footer = (
            footer[: max(0, inner_w - 6)]
            + "…"
        )

        footer_x = chat_w - len(footer) - 3

    min_fx = 2 + len(mode_str)

    if footer_x < min_fx:
        available = max(0, chat_w - min_fx - 3)

        if len(footer) > available:
            footer = (
                footer[: max(0, available - 1)]
                + "…"
            ) if available >= 1 else "…"

        footer_x = max(
            min_fx,
            chat_w - len(footer) - 3,
        )

    return mode_str, footer, footer_x


# ---------------------------------------------------------------------------
# Slash command autocomplete
# ---------------------------------------------------------------------------

SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/models", "Pick a model from the active provider"),
    ("/connect", "Connect a local or cloud provider"),
    ("/intel", "Set intelligence level"),
    ("/status", "Show current provider, model, mode and workspace"),
    ("/clear", "Clear the conversation"),
    ("/history", "Show recent conversation history"),
    ("/experiences", "Show recent learned experiences"),
    ("/tasks", "Show saved task state"),
    ("/check", "Run the A.S.C.S. system checks"),
    ("/help", "Show commands and keyboard controls"),
    ("/quit", "Exit A.S.C.S."),
)


def slash_menu_text() -> str:
    """Return the full slash-command list."""
    lines = ["Commands:"]

    for name, description in SLASH_COMMANDS:
        lines.append(f"  {name} — {description}")

    return "\n".join(lines)


def slash_suggestions(text: str) -> list[tuple[str, str]]:
    """Return commands matching the current slash input.

    Autocomplete is active only while the input is a single slash command
    without arguments. Once whitespace is entered, normal command editing
    takes over.
    """
    value = text.strip()

    if not value.startswith("/"):
        return []

    if " " in value or "\t" in value:
        return []

    query = value[1:].lower()

    if not query:
        return list(SLASH_COMMANDS)

    return [
        (command, description)
        for command, description in SLASH_COMMANDS
        if command[1:].lower().startswith(query)
    ]


def slash_completion(text: str, selected: int) -> str | None:
    """Return the selected command for the current slash input."""
    suggestions = slash_suggestions(text)

    if not suggestions:
        return None

    selected = max(
        0,
        min(selected, len(suggestions) - 1),
    )

    return suggestions[selected][0]


def parse_slash_command(
    text: str,
) -> tuple[str, list[str]]:
    """Parse '/intel high' into ('intel', ['high'])."""
    value = text.strip()

    if not value.startswith("/"):
        return "", []

    parts = value[1:].split()

    if not parts:
        return "", []

    return parts[0].lower(), parts[1:]


def _is_local_url(url: str) -> bool:
    """Return True for local endpoints."""
    host = (url or "").strip().lower()

    if "://" in host:
        host = (
            host.split("://", 1)[1]
            .split("/", 1)[0]
            .split("@")[-1]
        )

    host = host.split(":", 1)[0]

    return (
        host in ("localhost", "127.0.0.1", "::1")
        or host.endswith(".local")
    )


def validate_connect_inputs(
    provider: str,
    base_url: str,
    api_key: str,
) -> str | None:
    """Validate provider connection input."""
    from agent.models.providers import API_KEY_ENVS

    provider_name = (provider or "").strip().lower()
    base = (base_url or "").strip()

    if not base:
        return "Base URL is required."

    low = base.lower()

    if not (
        low.startswith("http://")
        or low.startswith("https://")
    ):
        return "Base URL must start with http:// or https://."

    if (
        provider_name != "ollama"
        and not api_key.strip()
        and not _is_local_url(base)
    ):
        env_name = (
            API_KEY_ENVS.get(provider_name)
            or "API_KEY"
        )

        return (
            f"API key is required for {provider_name} — "
            f"enter it or set {env_name}."
        )

    return None


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

class TuiApp:
    """Stateful real curses application."""

    def __init__(
        self,
        config: AgentConfig,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.client = client

        configured_mode = config.mode.upper()

        self.mode = (
            configured_mode
            if configured_mode in MODE_ORDER
            else "AUTO"
        )

        if self.mode == "SAFE":
            self.mode = "AUTO"

        self.provider = config.provider
        self.model = config.model
        self.intelligence = config.intelligence
        self.theme = config.theme

        self.input_text = ""
        self.cursor_pos = 0

        self.messages: list[dict[str, str]] = []
        self.history: list[str] = []

        self.status_msg = (
            "TAB mode  ·  / commands  ·  Enter send  ·  "
            "Ctrl+C cancel  ·  Esc quit"
        )

        self.should_quit = False

        self._hub = None
        self._runner = None

        self._scroll_offset = 0
        self._pending_status = ""
        self._last_event_count = 0

        # Slash autocomplete state.
        self._slash_selection = 0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def cycle_mode(self) -> None:
        self.mode = next_mode(self.mode)

        try:
            save_tui_state({"mode": self.mode})
        except Exception:
            pass

        self.status_msg = f"Mode → {self.mode}"

    def set_intelligence(self, level: str) -> str:
        lvl = validate_intel(level)

        self.intelligence = lvl

        n_ctx, n_pred, context_budget, _ = (
            intelligence_values(lvl)
        )

        try:
            save_tui_state(
                {
                    "intelligence": lvl,
                    "num_ctx": n_ctx,
                    "num_predict": n_pred,
                    "context_budget_chars": context_budget,
                }
            )
        except Exception:
            pass

        return (
            f"Intelligence → {lvl} "
            f"({n_ctx}/{n_pred}, budget {context_budget})"
        )

    def set_provider_model(
        self,
        provider: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        self.provider = provider

        if model:
            self.model = model

        data: dict[str, Any] = {
            "provider": provider,
            "model": self.model,
        }

        if base_url:
            data[
                "ollama_base_url"
                if provider == "ollama"
                else f"{provider}_base_url"
            ] = base_url

        try:
            save_tui_state(data)
        except Exception:
            pass

    def _live_config(
        self,
        workspace: Path | None = None,
    ) -> AgentConfig:
        """Build configuration from current UI state."""
        overrides: dict[str, Any] = {
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "intelligence": self.intelligence,
            "theme": self.theme,
            "workspace": (
                workspace
                if workspace is not None
                else self.config.workspace
            ),
        }

        try:
            return load_config(**overrides)
        except Exception:
            return self.config

    def _add_message(
        self,
        role: str,
        content: str,
    ) -> None:
        if not content:
            return

        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

        self.history.append(
            content
            if role == "user"
            else f"{role}: {content}"
        )

        self._scroll_offset = 0

    def _add_system(self, content: str) -> None:
        self._add_message("system", content)

    # ------------------------------------------------------------------
    # Slash autocomplete state
    # ------------------------------------------------------------------

    def _slash_active(self) -> bool:
        """Return True when live slash autocomplete should be displayed."""
        if not self.input_text.startswith("/"):
            return False

        if " " in self.input_text or "\t" in self.input_text:
            return False

        return True

    def _slash_items(self) -> list[tuple[str, str]]:
        """Return current autocomplete suggestions."""
        if not self._slash_active():
            return []

        return slash_suggestions(self.input_text)

    def _reset_slash_selection(self) -> None:
        self._slash_selection = 0

    def _clamp_slash_selection(self) -> None:
        items = self._slash_items()

        if not items:
            self._slash_selection = 0
            return

        self._slash_selection = max(
            0,
            min(
                self._slash_selection,
                len(items) - 1,
            ),
        )

    def _move_slash_selection(self, delta: int) -> bool:
        """Move slash selection. Returns whether autocomplete is active."""
        items = self._slash_items()

        if not items:
            return False

        self._slash_selection = (
            self._slash_selection + delta
        ) % len(items)

        return True

    def _complete_slash(self) -> bool:
        """Complete the highlighted slash command."""
        command = slash_completion(
            self.input_text,
            self._slash_selection,
        )

        if not command:
            return False

        self.input_text = command
        self.cursor_pos = len(command)
        self._reset_slash_selection()

        self.status_msg = (
            f"{command} selected — Enter to run"
        )

        return True

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------

    def _ensure_hub_runner(self):
        if self._hub is None:
            try:
                from agent.web import EventHub
            except Exception:
                return None, None

            self._hub = EventHub()

        return self._hub, None

    def _is_busy(self) -> bool:
        if self._runner is None:
            return False

        try:
            return bool(self._runner.busy)
        except Exception:
            return False

    def _start_task(self, text: str) -> bool:
        """Start a real AgentLoop task."""
        if self._is_busy():
            self._add_system(
                "Already running — press Ctrl+C to cancel."
            )
            return False

        try:
            live_cfg = self._live_config()
        except Exception as exc:
            self._add_system(f"Config error: {exc}")
            return False

        if self._hub is None:
            from agent.web import EventHub

            self._hub = EventHub()

        client = self.client

        if client is None:
            try:
                from agent.models.client import OllamaClient

                client = OllamaClient(
                    base_url=live_cfg.ollama_base_url,
                    model=live_cfg.model,
                    request_timeout=live_cfg.request_timeout,
                    keep_alive=live_cfg.keep_alive,
                    num_ctx=live_cfg.num_ctx,
                    num_predict=live_cfg.num_predict,
                )

            except Exception as exc:
                self._add_system(
                    f"Ollama client error: {exc}"
                )
                return False

        try:
            from agent.workspace import Workspace

            workspace = Workspace(live_cfg.workspace)

        except Exception as exc:
            self._add_system(
                f"Workspace error: {exc}"
            )
            return False

        try:
            from agent.web import TaskRunner

            self._runner = TaskRunner(
                live_cfg,
                client,
                workspace,
                self._hub,
            )

            started = self._runner.start(
                text,
                mode=live_cfg.mode,
            )

            if not started:
                self._add_system(
                    "Task already running."
                )
                return False

            self._add_message("user", text)

            self.status_msg = (
                f"Running ({live_cfg.mode}) — "
                "Ctrl+C to cancel"
            )

            self._last_event_count = len(
                self._hub.history()
            )

            return True

        except Exception as exc:
            self._add_system(
                f"Failed to start task: {exc}"
            )
            return False

    def _poll_runner(self) -> None:
        """Drain backend events into the conversation."""
        if self._hub is None:
            return

        try:
            events = self._hub.history()
        except Exception:
            return

        if len(events) <= self._last_event_count:
            if self._runner is not None:
                try:
                    if (
                        not self._runner.busy
                        and self._runner.result is not None
                    ):
                        result = self._runner.result

                        summary = (
                            getattr(result, "summary", "")
                            or getattr(result, "error", "")
                            or ""
                        )

                        status = getattr(
                            result,
                            "status",
                            "",
                        )

                        if status in (
                            "completed",
                            "COMPLETE",
                        ):
                            if summary:
                                self._add_message(
                                    "assistant",
                                    summary,
                                )

                            self.status_msg = (
                                "Completed — ready"
                            )

                        elif status in (
                            "failed",
                            "FAILED",
                        ):
                            self._add_system(
                                f"Failed: {summary}"
                            )

                            self.status_msg = (
                                "Failed — ready"
                            )

                        elif status in (
                            "cancelled",
                            "CANCELLED",
                        ):
                            self._add_system(
                                "Cancelled."
                            )

                            self.status_msg = (
                                "Cancelled — ready"
                            )

                        elif summary:
                            self._add_message(
                                "assistant",
                                summary,
                            )

                        self._runner = None

                except Exception:
                    pass

            return

        new_events = events[
            self._last_event_count:
        ]

        self._last_event_count = len(events)

        for event in new_events:
            try:
                if isinstance(event, dict):
                    event_type = (
                        event.get("type", "")
                        or ""
                    )

                    message = (
                        event.get("message", "")
                        or ""
                    )

                    tool = (
                        event.get("tool", "")
                        or ""
                    )

                    output = (
                        event.get("output", "")
                        or ""
                    )

                    target = (
                        event.get("target", "")
                        or ""
                    )

                else:
                    event_type = (
                        getattr(event, "type", "")
                        or ""
                    )

                    message = (
                        getattr(event, "message", "")
                        or ""
                    )

                    tool = (
                        getattr(event, "tool", "")
                        or ""
                    )

                    output = (
                        getattr(event, "output", "")
                        or ""
                    )

                    target = (
                        getattr(event, "target", "")
                        or ""
                    )

                if event_type in (
                    "agent_started",
                    "status",
                ):
                    if message:
                        self.status_msg = message

                elif event_type in (
                    "thinking",
                    "activity",
                ):
                    if message:
                        self.status_msg = message

                elif event_type == "model_started":
                    self.status_msg = "Thinking…"

                elif event_type == "model_completed":
                    self.status_msg = "Processing…"

                elif event_type == "tool_started":
                    if tool:
                        self._add_system(
                            f"→ {tool}"
                            + (
                                f"  {target}"
                                if target
                                else ""
                            )
                        )

                elif event_type == "tool_completed":
                    if tool:
                        self.status_msg = (
                            f"Finished {tool}"
                        )

                elif event_type == "command_output":
                    value = output or message

                    if value:
                        if len(value) > 800:
                            value = (
                                value[:800]
                                + "…"
                            )

                        self._add_system(
                            value.strip()
                        )

                elif event_type == "file_written":
                    if target:
                        self._add_system(
                            f"Wrote {target}"
                        )

                elif event_type == "patch_applied":
                    if target:
                        self._add_system(
                            f"Patched {target}"
                        )

                elif event_type in (
                    "agent_completed",
                    "task_completed",
                ):
                    if message:
                        self._add_message(
                            "assistant",
                            message,
                        )

                elif event_type == "agent_error":
                    if message:
                        self._add_system(
                            f"Error: {message}"
                        )

                elif event_type == "task_failed":
                    if message:
                        self._add_system(
                            f"Task failed: {message}"
                        )

                elif event_type == "task_plan":
                    if message:
                        self._add_system(message)

            except Exception:
                continue

        if self._runner is not None:
            try:
                if (
                    not self._runner.busy
                    and self._runner.result is not None
                ):
                    result = self._runner.result

                    summary = (
                        getattr(
                            result,
                            "summary",
                            "",
                        )
                        or ""
                    )

                    error = (
                        getattr(
                            result,
                            "error",
                            "",
                        )
                        or ""
                    )

                    status = getattr(
                        result,
                        "status",
                        "",
                    )

                    if (
                        error
                        and status
                        not in (
                            "completed",
                            "COMPLETE",
                        )
                    ):
                        self._add_system(
                            f"{status}: {error}"
                        )

                        self.status_msg = (
                            f"{status} — ready"
                        )

                    elif (
                        summary
                        and not any(
                            message["content"]
                            == summary
                            for message
                            in self.messages[-2:]
                        )
                    ):
                        self._add_message(
                            "assistant",
                            summary,
                        )

                        self.status_msg = (
                            "Completed — ready"
                        )

                    else:
                        self.status_msg = (
                            "Ready — TAB mode  |  "
                            "/ commands"
                        )

                    self._runner = None

            except Exception:
                pass

    def _cancel_running(self) -> None:
        if self._runner is None:
            self.status_msg = "Nothing to cancel."
            return

        try:
            self._runner.cancel()

            self.status_msg = "Cancelling…"

            self._add_system(
                "Cancelled by user."
            )

        except Exception as exc:
            self.status_msg = (
                f"Cancel failed: {exc}"
            )

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------

    def _init_colors(self, stdscr) -> None:
        if not HAS_CURSES or curses is None:
            return

        try:
            curses.use_default_colors()
            curses.curs_set(1)
        except Exception:
            pass

        try:
            if not curses.has_colors():
                return

            curses.start_color()

            tc = theme_colors(self.theme)

            use_256 = (
                getattr(curses, "COLORS", 8)
                >= 256
            )

            if use_256:
                bg = tc["bg_idx"]
                fg = tc["fg_idx"]

                curses.init_pair(
                    1,
                    fg,
                    bg,
                )

                curses.init_pair(
                    2,
                    MODE_COLOR_IDX["PLAN"],
                    bg,
                )

                curses.init_pair(
                    3,
                    MODE_COLOR_IDX["BUILD"],
                    bg,
                )

                curses.init_pair(
                    4,
                    MODE_COLOR_IDX["AUTO"],
                    bg,
                )

                curses.init_pair(
                    5,
                    PINK_FG_IDX,
                    PINK_BG_IDX,
                )

                curses.init_pair(
                    6,
                    curses.COLOR_WHITE,
                    PINK_BG_IDX,
                )

                curses.init_pair(
                    7,
                    curses.COLOR_BLACK,
                    curses.COLOR_CYAN,
                )

            else:
                bg = (
                    curses.COLOR_BLACK
                    if tc["theme"] == "dark"
                    else curses.COLOR_WHITE
                )

                fg = (
                    curses.COLOR_WHITE
                    if tc["theme"] == "dark"
                    else curses.COLOR_BLACK
                )

                curses.init_pair(
                    1,
                    fg,
                    bg,
                )

                curses.init_pair(
                    2,
                    curses.COLOR_YELLOW,
                    bg,
                )

                curses.init_pair(
                    3,
                    curses.COLOR_BLUE,
                    bg,
                )

                curses.init_pair(
                    4,
                    curses.COLOR_RED,
                    bg,
                )

                curses.init_pair(
                    5,
                    curses.COLOR_BLACK,
                    curses.COLOR_MAGENTA,
                )

        except Exception:
            try:
                curses.init_pair(
                    1,
                    curses.COLOR_WHITE,
                    curses.COLOR_BLACK,
                )

                curses.init_pair(
                    2,
                    curses.COLOR_YELLOW,
                    curses.COLOR_BLACK,
                )

                curses.init_pair(
                    3,
                    curses.COLOR_BLUE,
                    curses.COLOR_BLACK,
                )

                curses.init_pair(
                    4,
                    curses.COLOR_RED,
                    curses.COLOR_BLACK,
                )

                curses.init_pair(
                    5,
                    curses.COLOR_BLACK,
                    curses.COLOR_MAGENTA,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Text wrapping
    # ------------------------------------------------------------------

    def _wrap_lines(
        self,
        text: str,
        width: int,
    ) -> list[str]:
        """Word-wrap text."""
        if width <= 0:
            return [text]

        lines: list[str] = []

        for paragraph in text.split("\n"):
            if not paragraph:
                lines.append("")
                continue

            while len(paragraph) > width:
                cut = paragraph.rfind(
                    " ",
                    0,
                    width,
                )

                if cut <= width // 2:
                    cut = width

                lines.append(
                    paragraph[:cut]
                )

                paragraph = paragraph[
                    cut:
                ].lstrip()

            lines.append(paragraph)

        return lines

    # ------------------------------------------------------------------
    # Renderer
    # ------------------------------------------------------------------

    def _draw_header(
        self,
        stdscr,
        h: int,
        w: int,
    ) -> None:
        """Draw compact OpenCode-style header."""
        if h < 1 or w < 1:
            return

        pair = (
            curses.color_pair(1)
            if curses.has_colors()
            else 0
        )

        title = "A.S.C.S."
        subtitle = "A Smart Coding System"

        try:
            stdscr.addstr(
                0,
                2,
                title,
                curses.A_BOLD | pair,
            )

            if w > 30:
                stdscr.addstr(
                    0,
                    12,
                    subtitle,
                    curses.A_DIM | pair,
                )

            mode_pair = {
                "PLAN": 2,
                "BUILD": 3,
                "AUTO": 4,
            }.get(self.mode, 1)

            mode_text = f"[ {self.mode} ]"

            mode_x = max(
                2,
                (w - len(mode_text)) // 2,
            )

            stdscr.addstr(
                0,
                mode_x,
                mode_text,
                curses.A_BOLD
                | (
                    curses.color_pair(mode_pair)
                    if curses.has_colors()
                    else 0
                ),
            )

            footer = format_model_footer(
                self.model,
                self.intelligence,
            )

            if len(footer) < w - 22:
                stdscr.addstr(
                    0,
                    max(
                        2,
                        w - len(footer) - 2,
                    ),
                    footer,
                    curses.A_DIM | pair,
                )

        except curses.error:
            pass

    def _render_message_lines(
        self,
        width: int,
    ) -> list[tuple[str, int]]:
        """Turn conversation state into display lines."""
        rendered: list[tuple[str, int]] = []

        if not self.messages:
            lines = [
                "A.S.C.S. ready.",
                "",
                "Describe a coding task to begin.",
                "Type / for commands.",
            ]

            for line in lines:
                rendered.append(
                    (
                        line,
                        curses.A_DIM,
                    )
                )

            return rendered

        for message in self.messages:
            role = message.get(
                "role",
                "",
            )

            content = message.get(
                "content",
                "",
            )

            if not content:
                continue

            if role == "user":
                prefix = "> "
                attr = curses.A_BOLD

            elif role == "assistant":
                prefix = "  "
                attr = 0

            elif role == "system":
                prefix = "· "
                attr = curses.A_DIM

            else:
                prefix = "  "
                attr = curses.A_DIM

            content_lines = content.split("\n")

            first = True

            for content_line in content_lines:
                if first:
                    line = prefix + content_line
                    first = False
                else:
                    line = (
                        "  "
                        + content_line
                    )

                wrapped = self._wrap_lines(
                    line,
                    max(10, width),
                )

                for wrapped_line in wrapped:
                    rendered.append(
                        (
                            wrapped_line,
                            attr,
                        )
                    )

            rendered.append(
                (
                    "",
                    0,
                )
            )

        if rendered and not rendered[-1][0]:
            rendered.pop()

        return rendered

    def _draw_conversation(
        self,
        stdscr,
        y: int,
        x: int,
        width: int,
        height: int,
    ) -> None:
        """Draw borderless conversation area."""
        if width <= 0 or height <= 0:
            return

        rendered = self._render_message_lines(
            width - 2
        )

        if len(rendered) > height:
            start = max(
                0,
                len(rendered)
                - height
                - self._scroll_offset,
            )

            visible = rendered[
                start:
                start + height
            ]

        else:
            visible = rendered

        for idx, (line, attr) in enumerate(
            visible
        ):
            if idx >= height:
                break

            try:
                stdscr.addstr(
                    y + idx,
                    x,
                    line[:width],
                    attr,
                )
            except curses.error:
                pass

    def _draw_input(
        self,
        stdscr,
        y: int,
        x: int,
        width: int,
    ) -> None:
        """Draw the bottom command line."""
        if width <= 0 or y < 0:
            return

        prompt = "> "

        available = max(
            1,
            width - len(prompt) - 1,
        )

        text = self.input_text
        cursor = self.cursor_pos

        if len(text) <= available:
            visible = text
            cursor_x = (
                len(prompt) + cursor
            )

        else:
            if cursor <= available:
                start = 0
            elif cursor >= len(text):
                start = len(text) - available
            else:
                start = cursor - (
                    available // 2
                )

            start = max(0, start)

            visible = text[
                start:
                start + available
            ]

            cursor_x = (
                len(prompt)
                + cursor
                - start
            )

            cursor_x = max(
                len(prompt),
                min(
                    cursor_x,
                    len(prompt)
                    + len(visible),
                ),
            )

        try:
            pair = (
                curses.color_pair(1)
                if curses.has_colors()
                else 0
            )

            stdscr.addstr(
                y,
                x,
                prompt,
                curses.A_BOLD | pair,
            )

            stdscr.addstr(
                y,
                x + len(prompt),
                visible,
                pair,
            )

            remaining = max(
                0,
                width
                - len(prompt)
                - len(visible),
            )

            if remaining:
                stdscr.addstr(
                    y,
                    x
                    + len(prompt)
                    + len(visible),
                    " " * remaining,
                    pair,
                )

            stdscr.move(
                y,
                min(
                    x
                    + cursor_x,
                    x + width - 1,
                ),
            )

        except curses.error:
            pass

    def _draw_slash_menu(
        self,
        stdscr,
        input_y: int,
        x: int,
        width: int,
    ) -> int:
        """Draw live slash-command autocomplete above the input.

        Returns the number of rows consumed by the menu.
        """
        items = self._slash_items()

        if not items:
            return 0

        self._clamp_slash_selection()

        h, w = stdscr.getmaxyx()

        available_width = max(
            20,
            min(width, w - x - 2),
        )

        # Keep the menu compact rather than becoming another giant UI box.
        max_visible = min(
            8,
            max(1, h - 6),
            len(items),
        )

        selected = self._slash_selection

        visible_start = max(
            0,
            selected - max_visible + 2,
        )

        visible_start = min(
            visible_start,
            max(0, len(items) - max_visible),
        )

        visible_items = items[
            visible_start:
            visible_start + max_visible
        ]

        menu_height = len(visible_items) + 1

        # Place menu immediately above input.
        menu_y = input_y - menu_height - 1

        if menu_y < 2:
            menu_y = 2

        try:
            # Subtle separator/header.
            label = "Commands"
            header = (
                f"  {label}"
                + " "
                * max(
                    0,
                    available_width - len(label) - 2,
                )
            )

            stdscr.addstr(
                menu_y,
                x,
                header[:available_width],
                curses.A_DIM,
            )

            for offset, (
                command,
                description,
            ) in enumerate(visible_items):
                index = visible_start + offset
                y = menu_y + 1 + offset

                selected_row = (
                    index == selected
                )

                command_width = min(
                    16,
                    max(10, available_width // 3),
                )

                command_text = command.ljust(
                    command_width
                )

                remaining_width = max(
                    1,
                    available_width
                    - command_width
                    - 4,
                )

                desc_text = description[
                    :remaining_width
                ]

                line = (
                    "› "
                    + command_text
                    + "  "
                    + desc_text
                )

                line = line[:available_width]

                if selected_row:
                    attr = (
                        curses.color_pair(5)
                        | curses.A_BOLD
                        if curses.has_colors()
                        else curses.A_REVERSE
                    )

                    padded = line.ljust(
                        available_width
                    )

                    stdscr.addstr(
                        y,
                        x,
                        padded[:available_width],
                        attr,
                    )

                else:
                    stdscr.addstr(
                        y,
                        x,
                        line,
                        curses.A_DIM,
                    )

            # Autocomplete hint.
            hint_y = menu_y + menu_height

            if hint_y < input_y:
                hint = (
                    "↑/↓ select  ·  Tab complete  ·  Enter run"
                )

                stdscr.addstr(
                    hint_y,
                    x,
                    hint[:available_width],
                    curses.A_DIM,
                )

        except curses.error:
            return 0

        return menu_height + 1

    def _draw_status(
        self,
        stdscr,
        y: int,
        width: int,
    ) -> None:
        """Draw minimal controls/status footer."""
        if y < 0 or width <= 0:
            return

        status = self.status_msg

        if self._is_busy():
            status = "● " + status

        try:
            pair = (
                curses.color_pair(1)
                if curses.has_colors()
                else 0
            )

            stdscr.addstr(
                y,
                2,
                status[: max(0, width - 4)],
                curses.A_DIM | pair,
            )

        except curses.error:
            pass

    def _draw(self, stdscr) -> None:
        """Render the complete terminal interface."""
        if not HAS_CURSES or curses is None:
            return

        stdscr.erase()

        h, w = stdscr.getmaxyx()

        if h <= 0 or w <= 0:
            return

        try:
            if curses.has_colors():
                stdscr.bkgd(
                    " ",
                    curses.color_pair(1),
                )

                stdscr.erase()

        except Exception:
            pass

        geometry = calc_chatbox_geometry(
            h,
            w,
        )

        tier = geometry.get(
            "tier",
            "normal",
        )

        # --------------------------------------------------------------
        # Extremely small
        # --------------------------------------------------------------

        if geometry.get("too_small"):
            try:
                msg1 = "Terminal too small"
                msg2 = (
                    f"Minimum: "
                    f"{MIN_TERM_W}x{MIN_TERM_H}"
                )

                stdscr.addstr(
                    max(0, h // 2 - 1),
                    max(
                        0,
                        (w - len(msg1))
                        // 2,
                    ),
                    msg1,
                    curses.A_BOLD,
                )

                stdscr.addstr(
                    max(0, h // 2),
                    max(
                        0,
                        (w - len(msg2))
                        // 2,
                    ),
                    msg2,
                    curses.A_DIM,
                )

                stdscr.refresh()

            except curses.error:
                pass

            return

        # --------------------------------------------------------------
        # Minimised
        # --------------------------------------------------------------

        if geometry["is_minimised"]:
            try:
                self._draw_header(
                    stdscr,
                    h,
                    w,
                )

                message = (
                    "Resize terminal for full interface"
                    if tier == "minimised"
                    else "Terminal too small"
                )

                stdscr.addstr(
                    max(1, h // 2),
                    max(
                        0,
                        (w - len(message))
                        // 2,
                    ),
                    message,
                    curses.A_DIM,
                )

                self._draw_input(
                    stdscr,
                    max(1, h - 2),
                    2,
                    max(1, w - 4),
                )

                self._draw_status(
                    stdscr,
                    h - 1,
                    w,
                )

                stdscr.refresh()

            except curses.error:
                pass

            return

        # --------------------------------------------------------------
        # Full interface
        # --------------------------------------------------------------

        self._draw_header(
            stdscr,
            h,
            w,
        )

        # Header separator.
        try:
            if w > 2:
                stdscr.addstr(
                    1,
                    2,
                    "─" * max(0, w - 4),
                    curses.A_DIM,
                )
        except curses.error:
            pass

        # Reserve the bottom area for input, status and autocomplete.
        autocomplete_active = bool(
            self._slash_items()
        )

        input_separator_y = h - 4
        input_y = h - 3

        if autocomplete_active:
            menu_count = min(
                8,
                len(self._slash_items()),
            )

            # Keep enough conversation space visible.
            menu_height = menu_count + 2

            input_separator_y = h - 4

            conversation_height = max(
                3,
                h
                - 8
                - menu_height,
            )

        else:
            conversation_height = max(
                1,
                h - 8,
            )

        conversation_y = 3

        conversation_x = 2
        conversation_width = max(
            1,
            w - 4,
        )

        self._draw_conversation(
            stdscr,
            conversation_y,
            conversation_x,
            conversation_width,
            conversation_height,
        )

        # Input separator.
        try:
            if w > 2:
                stdscr.addstr(
                    input_separator_y,
                    2,
                    "─" * max(0, w - 4),
                    curses.A_DIM,
                )
        except curses.error:
            pass

        # Slash autocomplete lives above the input line.
        if autocomplete_active:
            self._draw_slash_menu(
                stdscr,
                input_y,
                2,
                max(1, w - 4),
            )

        self._draw_input(
            stdscr,
            input_y,
            2,
            max(1, w - 4),
        )

        # Footer.
        self._draw_status(
            stdscr,
            h - 1,
            w,
        )

        try:
            stdscr.noutrefresh()
            curses.doupdate()
        except Exception:
            try:
                stdscr.refresh()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _handle_input_key(
        self,
        ch: int,
        stdscr,
    ) -> bool:
        """Handle editing keys."""
        if ch in (
            10,
            13,
            curses.KEY_ENTER
            if HAS_CURSES and curses
            else 10,
        ):
            return True

        if ch in (
            8,
            127,
            curses.KEY_BACKSPACE
            if HAS_CURSES and curses
            else 127,
            263,
        ):
            if self.cursor_pos > 0:
                self.input_text = (
                    self.input_text[
                        : self.cursor_pos - 1
                    ]
                    + self.input_text[
                        self.cursor_pos:
                    ]
                )

                self.cursor_pos -= 1

            elif self.input_text:
                self.input_text = (
                    self.input_text[:-1]
                )

                self.cursor_pos = len(
                    self.input_text
                )

            self._reset_slash_selection()

            return False

        if ch == 330 or (
            HAS_CURSES
            and ch == curses.KEY_DC
        ):
            if (
                0 <= self.cursor_pos
                < len(self.input_text)
            ):
                self.input_text = (
                    self.input_text[
                        : self.cursor_pos
                    ]
                    + self.input_text[
                        self.cursor_pos + 1:
                    ]
                )

            self._reset_slash_selection()

            return False

        if ch == 260 or (
            HAS_CURSES
            and ch == curses.KEY_LEFT
        ):
            self.cursor_pos = max(
                0,
                self.cursor_pos - 1,
            )
            return False

        if ch == 261 or (
            HAS_CURSES
            and ch == curses.KEY_RIGHT
        ):
            self.cursor_pos = min(
                len(self.input_text),
                self.cursor_pos + 1,
            )
            return False

        if ch == 262 or (
            HAS_CURSES
            and ch == curses.KEY_HOME
        ):
            self.cursor_pos = 0
            return False

        if ch == 360 or (
            HAS_CURSES
            and ch == curses.KEY_END
        ):
            self.cursor_pos = len(
                self.input_text
            )
            return False

        if ch == 1:
            self.cursor_pos = 0
            return False

        if ch == 5:
            self.cursor_pos = len(
                self.input_text
            )
            return False

        if 32 <= ch <= 126:
            character = chr(ch)

            self.input_text = (
                self.input_text[
                    : self.cursor_pos
                ]
                + character
                + self.input_text[
                    self.cursor_pos:
                ]
            )

            self.cursor_pos += 1
            self._reset_slash_selection()

        return False

    def _handle_enter(
        self,
        stdscr,
    ) -> None:
        """Execute the current input or highlighted slash command."""
        text = self.input_text.strip()

        if not text:
            return

        # If slash autocomplete is active, Enter uses the highlighted
        # command. This makes "/" + Enter deterministic instead of
        # inserting a useless "Commands:" message into chat.
        if self._slash_active():
            items = self._slash_items()

            if items:
                command = slash_completion(
                    self.input_text,
                    self._slash_selection,
                )

                if command:
                    text = command

        self.input_text = ""
        self.cursor_pos = 0
        self._reset_slash_selection()

        if text.startswith("/"):
            self._handle_slash(
                stdscr,
                text,
            )

        else:
            self._start_task(text)

    def _handle_string_key(
        self,
        key: str,
        stdscr,
    ) -> None:
        """Handle a string returned by curses.get_wch()."""
        if key == "\t":
            if self._slash_active() and self._slash_items():
                self._complete_slash()
            else:
                self.cycle_mode()

            return

        if key in ("\n", "\r"):
            self._handle_enter(stdscr)
            return

        if key == "\x1b":
            if self._is_busy():
                self._cancel_running()
            else:
                self.should_quit = True

            return

        if key == "\x03":
            if self._is_busy():
                self._cancel_running()

            else:
                self.input_text = ""
                self.cursor_pos = 0
                self._reset_slash_selection()

                self.status_msg = (
                    "Input cleared — "
                    "Ctrl+C again to quit, "
                    "/quit to exit"
                )

            return

        if key == "\x04":
            self.should_quit = True
            return

        if key in ("\x7f", "\b"):
            self._handle_input_key(
                127,
                stdscr,
            )
            return

        if (
            len(key) == 1
            and (
                32
                <= ord(key)
                <= 126
                or ord(key) > 127
            )
        ):
            self.input_text = (
                self.input_text[
                    : self.cursor_pos
                ]
                + key
                + self.input_text[
                    self.cursor_pos:
                ]
            )

            self.cursor_pos += 1
            self._reset_slash_selection()

    def _handle_integer_key(
        self,
        key: int,
        stdscr,
    ) -> None:
        """Handle an integer key returned by curses.get_wch()."""
        if key == 9:
            if self._slash_active() and self._slash_items():
                self._complete_slash()
            else:
                self.cycle_mode()

            return

        if (
            HAS_CURSES
            and key == curses.KEY_RESIZE
        ):
            return

        if key in (
            10,
            13,
            curses.KEY_ENTER
            if HAS_CURSES
            else 10,
        ):
            self._handle_enter(stdscr)
            return

        if key == 3:
            if self._is_busy():
                self._cancel_running()

            else:
                self.input_text = ""
                self.cursor_pos = 0
                self._reset_slash_selection()

                self.status_msg = (
                    "Input cleared — "
                    "Ctrl+C again to quit, "
                    "/quit to exit"
                )

            return

        if key == 4:
            self.should_quit = True
            return

        # Slash autocomplete owns the arrow keys while active.
        if self._slash_active() and self._slash_items():
            if (
                HAS_CURSES
                and key == curses.KEY_UP
            ) or key == 259:
                self._move_slash_selection(-1)
                return

            if (
                HAS_CURSES
                and key == curses.KEY_DOWN
            ) or key == 258:
                self._move_slash_selection(1)
                return

        # Otherwise arrows scroll conversation vertically.
        if (
            HAS_CURSES
            and key == curses.KEY_UP
        ):
            self._scroll_offset = min(
                self._scroll_offset + 1,
                max(
                    0,
                    len(self.messages) * 2,
                ),
            )
            return

        if (
            HAS_CURSES
            and key == curses.KEY_DOWN
        ):
            self._scroll_offset = max(
                0,
                self._scroll_offset - 1,
            )
            return

        self._handle_input_key(
            key,
            stdscr,
        )

    # ------------------------------------------------------------------
    # Pickers
    # ------------------------------------------------------------------

    def _run_picker(
        self,
        stdscr,
        provider_models: dict[str, list[str]],
        active_only: bool = False,
    ) -> tuple[str, str] | None:
        if not HAS_CURSES or curses is None:
            return None

        items = (
            build_scoped_picker_items(
                provider_models,
                self.provider,
            )
            if active_only
            else build_picker_items(
                provider_models
            )
        )

        if not items:
            return None

        selected = 0

        for index, item in enumerate(items):
            if (
                item.is_provider_header
                and item.provider
                == self.provider
            ):
                selected = index
                break

        title = (
            " Select model "
            "· Enter confirm · Esc cancel "
        )

        while True:
            h, w = stdscr.getmaxyx()

            picker_h = min(
                len(items) + 4,
                max(5, h - 4),
            )

            picker_w = min(
                64,
                max(30, w - 4),
            )

            picker_y = max(
                0,
                (h - picker_h) // 2,
            )

            picker_x = max(
                0,
                (w - picker_w) // 2,
            )

            try:
                win = curses.newwin(
                    picker_h,
                    picker_w,
                    picker_y,
                    picker_x,
                )

                if curses.has_colors():
                    win.bkgd(
                        " ",
                        curses.color_pair(1),
                    )

                win.box()

                win.addstr(
                    0,
                    max(
                        1,
                        (picker_w - len(title))
                        // 2,
                    ),
                    title[
                        : max(0, picker_w - 2)
                    ],
                    curses.A_BOLD,
                )

                visible_height = max(
                    1,
                    picker_h - 3,
                )

                visible_start = max(
                    0,
                    selected
                    - visible_height // 2,
                )

                visible_end = min(
                    len(items),
                    visible_start
                    + visible_height,
                )

                if (
                    visible_end
                    - visible_start
                    < visible_height
                ):
                    visible_start = max(
                        0,
                        visible_end
                        - visible_height,
                    )

                for index in range(
                    visible_start,
                    visible_end,
                ):
                    item = items[index]

                    y = (
                        1
                        + index
                        - visible_start
                    )

                    if item.is_provider_header:
                        text = (
                            f"  {item.provider}"
                        )

                        attr = curses.A_BOLD

                    else:
                        text = (
                            f"    {item.label}"
                        )

                        attr = 0

                    text = text[
                        : max(
                            0,
                            picker_w - 2,
                        )
                    ].ljust(
                        max(
                            0,
                            picker_w - 2,
                        )
                    )

                    if index == selected:
                        highlight = (
                            curses.color_pair(5)
                            | curses.A_BOLD
                            if curses.has_colors()
                            else curses.A_REVERSE
                        )

                        win.addstr(
                            y,
                            1,
                            text,
                            highlight,
                        )

                    else:
                        win.addstr(
                            y,
                            1,
                            text,
                            attr,
                        )

                win.addstr(
                    picker_h - 1,
                    2,
                    "↑/↓ move · Enter select · Esc cancel"
                    [: max(0, picker_w - 4)],
                    curses.A_DIM,
                )

                win.noutrefresh()
                curses.doupdate()

            except curses.error:
                pass

            try:
                key = stdscr.getch()
            except Exception:
                return None

            if key == 27:
                return None

            if key in (
                10,
                13,
                curses.KEY_ENTER,
            ):
                chosen = items[selected]

                if chosen.is_provider_header:
                    return (
                        chosen.provider,
                        "",
                    )

                return (
                    chosen.provider,
                    chosen.label,
                )

            if key in (
                curses.KEY_UP,
                259,
            ):
                selected = max(
                    0,
                    selected - 1,
                )

            elif key in (
                curses.KEY_DOWN,
                258,
            ):
                selected = min(
                    len(items) - 1,
                    selected + 1,
                )

            elif key == 9:
                selected = min(
                    len(items) - 1,
                    selected + 1,
                )

            elif (
                HAS_CURSES
                and hasattr(curses, "KEY_RESIZE")
                and key == curses.KEY_RESIZE
            ):
                continue

    def _run_intel_picker(
        self,
        stdscr,
    ) -> str | None:
        if not HAS_CURSES or curses is None:
            return None

        h, w = stdscr.getmaxyx()

        picker_h = len(
            INTEL_DISPLAY_ORDER
        ) + 4

        picker_w = 42

        picker_h = min(
            picker_h,
            max(5, h - 4),
        )

        picker_w = min(
            picker_w,
            max(24, w - 4),
        )

        selected = 0

        for index, level in enumerate(
            INTEL_DISPLAY_ORDER
        ):
            if level == self.intelligence:
                selected = index
                break

        title = (
            " Intelligence "
            "· Enter confirm · Esc cancel "
        )

        while True:
            h, w = stdscr.getmaxyx()

            picker_y = max(
                0,
                (h - picker_h) // 2,
            )

            picker_x = max(
                0,
                (w - picker_w) // 2,
            )

            try:
                win = curses.newwin(
                    picker_h,
                    picker_w,
                    picker_y,
                    picker_x,
                )

                if curses.has_colors():
                    win.bkgd(
                        " ",
                        curses.color_pair(1),
                    )

                win.box()

                win.addstr(
                    0,
                    max(
                        1,
                        (picker_w - len(title))
                        // 2,
                    ),
                    title[
                        : max(0, picker_w - 2)
                    ],
                    curses.A_BOLD,
                )

                for index, level in enumerate(
                    INTEL_DISPLAY_ORDER
                ):
                    y = 1 + index

                    n_ctx, n_pred, _, _ = (
                        intelligence_values(level)
                    )

                    marker = (
                        "●"
                        if level
                        == self.intelligence
                        else " "
                    )

                    text = (
                        f" {marker} "
                        f"{level:<8} "
                        f"{n_ctx}/{n_pred}"
                    )

                    text = text[
                        : max(0, picker_w - 2)
                    ].ljust(
                        max(0, picker_w - 2)
                    )

                    if index == selected:
                        attr = (
                            curses.color_pair(5)
                            | curses.A_BOLD
                            if curses.has_colors()
                            else curses.A_REVERSE
                        )

                    else:
                        attr = 0

                    win.addstr(
                        y,
                        1,
                        text,
                        attr,
                    )

                win.addstr(
                    picker_h - 1,
                    2,
                    "↑/↓ move · Enter select · Esc cancel"
                    [: max(0, picker_w - 4)],
                    curses.A_DIM,
                )

                win.noutrefresh()
                curses.doupdate()

            except curses.error:
                pass

            try:
                key = stdscr.getch()
            except Exception:
                return None

            if key == 27:
                return None

            if key in (
                10,
                13,
                curses.KEY_ENTER,
            ):
                return INTEL_DISPLAY_ORDER[
                    selected
                ]

            if key in (
                curses.KEY_UP,
                259,
            ):
                selected = max(
                    0,
                    selected - 1,
                )

            elif key in (
                curses.KEY_DOWN,
                258,
            ):
                selected = min(
                    len(INTEL_DISPLAY_ORDER) - 1,
                    selected + 1,
                )

            elif key == 9:
                selected = min(
                    len(INTEL_DISPLAY_ORDER) - 1,
                    selected + 1,
                )

    # ------------------------------------------------------------------
    # Provider connection
    # ------------------------------------------------------------------

    def _do_connect(self, stdscr) -> None:
        from agent.models.providers import (
            DEFAULT_BASE_URLS,
            API_KEY_ENVS,
            list_all_providers_with_models,
            list_models_for_provider,
        )

        self.status_msg = "Loading providers…"
        self._draw(stdscr)

        try:
            provider_models = (
                list_all_providers_with_models(
                    timeout=2,
                    use_cache=True,
                )
            )
        except Exception:
            provider_models = {
                provider: []
                for provider in PROVIDER_NAMES
            }

        result = self._run_picker(
            stdscr,
            provider_models,
        )

        if not result:
            return

        provider, _ = result

        h, w = stdscr.getmaxyx()

        dialog_h = 9
        dialog_w = min(
            64,
            max(20, w - 4),
        )

        dialog_y = max(
            0,
            (h - dialog_h) // 2,
        )

        dialog_x = max(
            0,
            (w - dialog_w) // 2,
        )

        base_url = DEFAULT_BASE_URLS.get(
            provider,
            "",
        )

        persisted = load_tui_state()

        providers = persisted.get(
            "providers",
            {},
        )

        if isinstance(providers, dict):
            saved = providers.get(
                provider,
                {},
            )

            if isinstance(saved, dict):
                if saved.get("base_url"):
                    base_url = str(
                        saved["base_url"]
                    )

        api_key = ""

        env_name = API_KEY_ENVS.get(
            provider
        )

        if env_name:
            api_key = os.environ.get(
                env_name,
                "",
            )

        fields = (
            1
            if provider == "ollama"
            else 2
        )

        labels = [
            "Base URL:",
            "API Key:",
        ]

        buffers = [
            base_url,
            api_key,
        ]

        field = 0
        error_message: str | None = None

        while True:
            h, w = stdscr.getmaxyx()

            dialog_w = min(
                64,
                max(20, w - 4),
            )

            dialog_x = max(
                0,
                (w - dialog_w) // 2,
            )

            dialog_y = max(
                0,
                (h - dialog_h) // 2,
            )

            try:
                win = curses.newwin(
                    dialog_h,
                    dialog_w,
                    dialog_y,
                    dialog_x,
                )

                if curses.has_colors():
                    win.bkgd(
                        " ",
                        curses.color_pair(1),
                    )

                win.box()

                title = (
                    f" Connect · {provider} "
                )

                win.addstr(
                    0,
                    max(
                        1,
                        (dialog_w - len(title))
                        // 2,
                    ),
                    title,
                    curses.A_BOLD,
                )

                for index in range(fields):
                    label_y = (
                        2
                        + index * 2
                    )

                    input_y = label_y + 1

                    win.addstr(
                        label_y,
                        2,
                        labels[index][
                            : max(
                                0,
                                dialog_w - 4,
                            )
                        ],
                        curses.A_DIM,
                    )

                    value = buffers[index]

                    if index == 1 and value:
                        display = "*" * len(value)
                    else:
                        display = value

                    width = max(
                        0,
                        dialog_w - 4,
                    )

                    display = display[
                        :width
                    ].ljust(width)

                    if index == field:
                        attr = (
                            curses.color_pair(5)
                            if curses.has_colors()
                            else curses.A_REVERSE
                        )
                    else:
                        attr = 0

                    win.addstr(
                        input_y,
                        2,
                        display,
                        attr,
                    )

                if error_message:
                    win.addstr(
                        dialog_h - 3,
                        2,
                        error_message[
                            : max(
                                0,
                                dialog_w - 4,
                            )
                        ],
                        curses.A_BOLD,
                    )

                win.addstr(
                    dialog_h - 2,
                    2,
                    "Enter confirm · Tab next · Esc cancel"
                    [: max(0, dialog_w - 4)],
                    curses.A_DIM,
                )

                win.noutrefresh()
                curses.doupdate()

            except curses.error:
                pass

            try:
                key = stdscr.getch()
            except Exception:
                return

            if key == 27:
                return

            if key == 9:
                field = (
                    field + 1
                ) % fields
                continue

            if key in (10, 13):
                new_base = buffers[0].strip()

                new_key = (
                    buffers[1].strip()
                    if fields > 1
                    else ""
                )

                validation = (
                    validate_connect_inputs(
                        provider,
                        new_base,
                        new_key,
                    )
                )

                if validation:
                    error_message = validation
                    self.status_msg = validation
                    continue

                try:
                    models = (
                        list_models_for_provider(
                            provider,
                            base_url=new_base,
                            api_key=(
                                new_key
                                or None
                            ),
                            timeout=5,
                            use_cache=False,
                        )
                    )

                except Exception as exc:
                    error_message = (
                        f"Connection failed: {exc}"
                    )
                    continue

                if not models:
                    error_message = (
                        "Connection failed: "
                        "no models returned."
                    )
                    continue

                try:
                    state = load_tui_state()

                    provider_state = state.get(
                        "providers",
                        {},
                    )

                    if not isinstance(
                        provider_state,
                        dict,
                    ):
                        provider_state = {}

                    provider_state[
                        provider
                    ] = {
                        "base_url": new_base
                    }

                    if (
                        new_key
                        and provider != "ollama"
                    ):
                        provider_state[
                            provider
                        ]["api_key"] = new_key

                    state[
                        "providers"
                    ] = provider_state

                    state[
                        "provider"
                    ] = provider

                    save_tui_state(state)

                    if env_name and new_key:
                        os.environ[
                            env_name
                        ] = new_key

                    self.provider = provider

                    self.status_msg = (
                        f"Connected to "
                        f"{provider} "
                        f"({len(models)} models)"
                    )

                    self._add_system(
                        self.status_msg
                    )

                except Exception as exc:
                    self.status_msg = (
                        f"Connect failed: {exc}"
                    )

                    self._add_system(
                        self.status_msg
                    )

                return

            if key in (
                curses.KEY_BACKSPACE,
                127,
                8,
                263,
            ):
                if buffers[field]:
                    buffers[field] = (
                        buffers[field][:-1]
                    )

            elif 32 <= key <= 126:
                buffers[field] += chr(key)

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def _handle_slash(
        self,
        stdscr,
        text: str,
    ) -> None:
        command, args = (
            parse_slash_command(text)
        )

        if not command:
            self.status_msg = (
                "Type / to see available commands"
            )
            return

        if command in (
            "help",
            "?",
        ):
            self._add_system(
                "Commands: "
                "/models  /connect  "
                "/intel [low|medium|high|xhigh|default]  "
                "/status  /clear  /history  "
                "/experiences  /tasks  /check  /quit\n"
                "Keys: TAB=mode  Enter=send  "
                "Esc=quit  Ctrl+C=cancel  "
                "Home/End  Arrows  Backspace/Delete  "
                "Up/Down scroll  "
                "Slash ↑/↓=select  Tab=complete"
            )

            self.status_msg = "Help shown"
            return

        if command in (
            "clear",
            "cls",
        ):
            self.messages.clear()
            self.history.clear()

            if self._hub is not None:
                try:
                    self._hub.clear()
                except Exception:
                    pass

            self._scroll_offset = 0

            self.status_msg = "Cleared"

            return

        if command in (
            "quit",
            "exit",
            "q",
        ):
            self.should_quit = True
            return

        if command == "intel":
            if not args:
                chosen = self._run_intel_picker(
                    stdscr
                )

                if chosen:
                    try:
                        message = (
                            self.set_intelligence(
                                chosen
                            )
                        )

                        self.status_msg = message
                        self._add_system(message)

                    except ValueError as exc:
                        self.status_msg = str(exc)
                        self._add_system(str(exc))

                else:
                    self.status_msg = (
                        "Intelligence selection cancelled"
                    )

                return

            try:
                message = (
                    self.set_intelligence(
                        args[0]
                    )
                )

                self.status_msg = message
                self._add_system(message)

            except ValueError as exc:
                self.status_msg = str(exc)
                self._add_system(str(exc))

            return

        if command == "models":
            from agent.models.providers import (
                list_all_providers_with_models,
            )

            try:
                provider_models = (
                    list_all_providers_with_models(
                        timeout=2,
                        use_cache=True,
                    )
                )

            except Exception:
                provider_models = {
                    provider: []
                    for provider
                    in PROVIDER_NAMES
                }

                self._add_system(
                    "Unable to load models."
                )

            active_models = (
                provider_models.get(
                    self.provider,
                    [],
                )
                or []
            )

            if not active_models:
                message = (
                    f"No models listed for "
                    f"{self.provider} — "
                    "try /connect first"
                )

                self.status_msg = message
                self._add_system(message)

                return

            result = self._run_picker(
                stdscr,
                provider_models,
                active_only=True,
            )

            if result:
                provider, model = result

                if model:
                    self.set_provider_model(
                        provider,
                        model,
                    )

                    self.status_msg = (
                        f"Model → "
                        f"{provider}/{model}"
                    )

                    self._add_system(
                        self.status_msg
                    )

                else:
                    self.set_provider_model(
                        provider,
                        "",
                    )

                    self.status_msg = (
                        f"Provider → {provider}"
                    )

                    self._add_system(
                        self.status_msg
                    )

            else:
                self.status_msg = (
                    "Model selection cancelled"
                )

            return

        if command == "connect":
            try:
                self._do_connect(stdscr)

            except Exception:
                self.status_msg = (
                    "Unable to connect provider."
                )

                self._add_system(
                    self.status_msg
                )

            return

        if command == "status":
            try:
                live = self._live_config()
                workspace = str(
                    live.workspace
                )

            except Exception:
                workspace = str(
                    self.config.workspace
                )

            connection = "unknown"

            try:
                from agent.models.client import OllamaClient

                client = (
                    self.client
                    or OllamaClient(
                        model=self.model
                    )
                )

                report = client.ensure_ready(
                    check_timeout=5,
                    prewarm=False,
                )

                if report.get("available"):
                    connection = "ready"
                else:
                    connection = (
                        f"model "
                        f"{self.model!r} "
                        "not installed"
                    )

                if not report.get("reachable"):
                    connection = (
                        "Ollama unreachable"
                    )

            except Exception as exc:
                connection = f"error: {exc}"

            self._add_system(
                f"Provider: {self.provider}\n"
                f"Model: {self.model}\n"
                f"Intelligence: {self.intelligence}\n"
                f"Mode: {self.mode}\n"
                f"Workspace: {workspace}\n"
                f"Connection: {connection}\n"
                f"Theme: {self.theme} "
                f"({detect_theme(self.theme)})\n"
                f"Busy: {self._is_busy()}"
            )

            self.status_msg = "Status shown"
            return

        if command == "history":
            if not self.messages:
                self._add_system(
                    "No history yet."
                )

            else:
                history = "\n".join(
                    f"{message['role']}: "
                    f"{message['content']}"
                    for message
                    in self.messages[-20:]
                )

                self._add_system(
                    history
                    or "No history"
                )

            return

        if command == "experiences":
            try:
                from agent.experience import (
                    ExperienceStore,
                )

                store = ExperienceStore()

                experiences = store.recent(
                    limit=5
                )

                if not experiences:
                    self._add_system(
                        "No experiences yet."
                    )

                else:
                    output = "\n".join(
                        f"- {experience.task[:80]} "
                        f"→ "
                        f"{'success' if experience.success else 'failed'} "
                        f"(score {experience.score:.2f})"
                        for experience
                        in experiences
                    )

                    self._add_system(output)

            except Exception as exc:
                self._add_system(
                    f"Experiences error: {exc}"
                )

            return

        if command == "tasks":
            try:
                from agent.context.project import (
                    ProjectStore,
                )

                store = ProjectStore(
                    self._live_config().workspace
                )

                graph = (
                    store.load_task_graph()
                )

                if not graph:
                    self._add_system(
                        "No saved tasks "
                        "(.ascs/task_state.json empty)."
                    )

                else:
                    tasks = list(
                        graph.tasks.values()
                    )[:10]

                    summary = (
                        f"Tasks: "
                        f"{len(graph.tasks)} — "
                        + ", ".join(
                            f"{task.id}:"
                            f"{task.status}"
                            for task in tasks
                        )
                    )

                    self._add_system(summary)

            except Exception as exc:
                self._add_system(
                    f"Tasks error: {exc}"
                )

            return

        if command == "check":
            try:
                from agent.doctor import doctor

                report = doctor(
                    workspace=str(
                        self._live_config()
                        .workspace
                    )
                )

                lines = [
                    f"{result.status} "
                    f"{result.name}: "
                    f"{result.message}"
                    for result in report.results
                ]

                self._add_system(
                    "\n".join(lines)
                )

            except Exception as exc:
                self._add_system(
                    f"Check failed: {exc}"
                )

            return

        self._add_system(
            f"Unknown command /{command} "
            "— type / for available commands"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_curses(self, stdscr) -> None:
        self._init_colors(stdscr)

        stdscr.keypad(True)
        stdscr.timeout(100)

        try:
            curses.cbreak()
        except Exception:
            pass

        try:
            curses.curs_set(1)
        except Exception:
            pass

        if not self.messages:
            self._add_system(
                f"A.S.C.S ready — "
                f"{self.model}"
                f"({self.intelligence}) "
                f"on {self.provider} — "
                "type / for commands"
            )

        self._draw(stdscr)

        while not self.should_quit:
            self._poll_runner()
            self._draw(stdscr)

            try:
                key = stdscr.get_wch()

                if isinstance(key, str):
                    self._handle_string_key(
                        key,
                        stdscr,
                    )

                else:
                    self._handle_integer_key(
                        key,
                        stdscr,
                    )

            except curses.error:
                continue

            except KeyboardInterrupt:
                if self._is_busy():
                    self._cancel_running()
                else:
                    self.should_quit = True
                    break

        if self._is_busy():
            try:
                self._cancel_running()
            except Exception:
                pass

    def run_fallback(self) -> int:
        """Fallback when curses is unavailable."""
        print(
            "A.S.C.S. TUI requires a real terminal.",
            file=sys.stderr,
        )

        print(
            "Run from PowerShell/Terminal:",
            file=sys.stderr,
        )

        print(
            "  .\\risa.cmd --tui",
            file=sys.stderr,
        )

        if not HAS_CURSES:
            print(
                "On Windows: "
                "pip install windows-curses",
                file=sys.stderr,
            )

        return 1


def run_tui(
    config: AgentConfig,
    client: Any | None = None,
    *,
    block: bool = True,
) -> int:
    """Run the real full-screen TUI."""
    del block  # retained for API compatibility

    app = TuiApp(
        config,
        client,
    )

    if not HAS_CURSES or curses is None:
        print(
            "curses not available on this platform.",
            file=sys.stderr,
        )

        print(
            "On Windows, install with: "
            "pip install windows-curses",
            file=sys.stderr,
        )

        return 1

    if (
        not sys.stdout.isatty()
        or not sys.stdin.isatty()
    ):
        print(
            "TUI requires a real terminal (TTY).",
            file=sys.stderr,
        )

        print(
            "Open a real terminal and run:",
            file=sys.stderr,
        )

        print(
            "  python -m agent --tui",
            file=sys.stderr,
        )

        return 1

    term = os.environ.get(
        "TERM",
        "",
    )

    if not term or term == "dumb":
        os.environ[
            "TERM"
        ] = "xterm-256color"

    try:
        return curses.wrapper(
            app.run_curses
        )

    except curses.error as exc:
        print(
            f"Curses error: {exc}",
            file=sys.stderr,
        )

        print(
            "Ensure terminal size >= "
            f"{MIN_TERM_W}x{MIN_TERM_H}.",
            file=sys.stderr,
        )

        return 1

    except KeyboardInterrupt:
        return 130

    finally:
        try:
            if HAS_CURSES and curses is not None:
                try:
                    curses.curs_set(1)
                except Exception:
                    pass

                try:
                    curses.echo()
                except Exception:
                    pass

                try:
                    curses.nocbreak()
                except Exception:
                    pass

        except Exception:
            pass