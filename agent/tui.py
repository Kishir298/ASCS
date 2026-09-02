"""Interactive TUI for A.S.C.S. — real terminal shell (OpenCode-inspired interaction).

Features:
  - TAB cycles Plan(orange) -> Build(blue) -> Auto(red)
  - /models  -> provider-aware model picker (bold provider, pink highlight)
  - /connect -> provider connector (local + cloud)
  - /intel   -> low/medium/high/xhigh/default  -> (num_ctx, num_predict, budget, level) + picker
  - Full-screen curses application, keyboard-first, responsive tiers, streaming
  - Real AgentLoop -> OllamaClient -> qwen3-coder:30b (no fake preview/queue)
  - Persistence via tui_state.json

Zero extra deps beyond stdlib curses (windows-curses on Windows).
"""

from __future__ import annotations

import os
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    AgentConfig,
    DEFAULT_INTELLIGENCE,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    INTELLIGENCE_LEVELS,
    INTELLIGENCE_MAP,
    PROVIDER_NAMES,
    THEMES,
    intelligence_values,
    load_config,
    load_tui_state,
    save_tui_state,
    tui_state_path,
)

# ---------------------------------------------------------------------------
# Constants (kept for test compatibility; HELLO_TEXT no longer rendered as demo)
# ---------------------------------------------------------------------------

HELLO_TEXT = "Hello, hello, hello, hello, hello, hello, hello, hello, hello"
HELLO_LEN = len(HELLO_TEXT)  # 61
MIN_CHATBOX_INNER_W = HELLO_LEN  # 61 — minimum guard only, layout is dynamic
MIN_CHATBOX_W = MIN_CHATBOX_INNER_W + 2  # 63 inc borders
MIN_CHATBOX_H = 5  # at least 3 content lines + 2 borders
MIN_TERM_W = 40
MIN_TERM_H = 10

MODE_ORDER = ("PLAN", "BUILD", "AUTO")
MODE_COLORS = {
    "PLAN": "orange",
    "BUILD": "blue",
    "AUTO": "red",
}
MODE_COLOR_IDX = {"PLAN": 208, "BUILD": 27, "AUTO": 196}
PINK_BG_IDX = 213
PINK_FG_IDX = 16

INTEL_CHOICES = ("low", "medium", "high", "xhigh", "default")
INTEL_DISPLAY_ORDER = ("default", "low", "medium", "high", "xhigh")

try:
    import curses  # type: ignore
    import curses.textpad  # noqa: F401

    HAS_CURSES = True
except Exception:  # pragma: no cover - Windows without windows-curses
    curses = None  # type: ignore[assignment]
    HAS_CURSES = False


# ---------------------------------------------------------------------------
# Helpers (testable without curses)
# ---------------------------------------------------------------------------

def next_mode(current: str) -> str:
    """Cycle Plan -> Build -> Auto -> Plan."""
    cur = (current or "").strip().upper()
    if cur not in MODE_ORDER:
        return "PLAN"
    idx = MODE_ORDER.index(cur)
    return MODE_ORDER[(idx + 1) % len(MODE_ORDER)]


def format_model_footer(model: str, intelligence: str) -> str:
    """Return 'model(intelligence)' as required."""
    return f"{model}({intelligence})"


def get_layout_tier(term_h: int, term_w: int) -> str:
    """Return responsive tier for given terminal size."""
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
    """Return geometry for chatbox given terminal size."""
    tier = get_layout_tier(term_h, term_w)
    is_min = 1 if tier in ("minimised", "extremely_small") else 0
    too_small = 1 if tier == "extremely_small" else 0
    if is_min:
        return {
            "chat_h": 0,
            "chat_w": 0,
            "chat_y": 0,
            "chat_x": 0,
            "is_minimised": 1,
            "inner_w": 0,
            "inner_h": 0,
            "tier": tier,  # type: ignore
            "too_small": too_small,  # type: ignore
        }
    if tier == "compact":
        side_margin = 0
        avail_w = term_w - side_margin * 2
    elif tier in ("large", "wide"):
        side_margin = 2
        avail_w = term_w - side_margin * 2
        max_chat_w = 110 if tier == "large" else 120
        avail_w = min(avail_w, max_chat_w)
    else:  # normal
        side_margin = 1
        avail_w = term_w - 2

    avail_h = term_h - 5
    chat_h = max(MIN_CHATBOX_H, min(avail_h, term_h - 5))
    if term_h >= 30:
        chat_h = min(chat_h, max(MIN_CHATBOX_H, term_h // 3 + 2))
    chat_w = max(MIN_CHATBOX_W, avail_w)
    chat_x = max(0, (term_w - chat_w) // 2)
    chat_y = 1
    inner_w = chat_w - 2
    inner_h = chat_h - 2
    return {
        "chat_h": chat_h,
        "chat_w": chat_w,
        "chat_y": chat_y,
        "chat_x": chat_x,
        "is_minimised": 0,
        "inner_w": inner_w,
        "inner_h": inner_h,
        "tier": tier,  # type: ignore
        "too_small": 0,  # type: ignore
    }


def is_minimised(term_h: int, term_w: int) -> bool:
    return bool(calc_chatbox_geometry(term_h, term_w)["is_minimised"])


def is_too_small(term_h: int, term_w: int) -> bool:
    """True when terminal is extremely small and guard message should show."""
    return get_layout_tier(term_h, term_w) == "extremely_small"


def detect_theme(config_theme: str) -> str:
    """Resolve 'auto' to 'light' or 'dark' based on env hints."""
    t = (config_theme or "auto").lower()
    if t in ("light", "dark"):
        return t
    if os.environ.get("WT_SESSION"):
        cfb = os.environ.get("COLORFGBG", "")
        if cfb:
            parts = cfb.replace(":", ";").split(";")
            if parts:
                try:
                    bg = int(parts[-1])
                    if 7 <= bg <= 15:
                        return "light"
                    if bg <= 6:
                        return "dark"
                except Exception:
                    pass
        return "dark"
    cfb = os.environ.get("COLORFGBG", "")
    if cfb:
        parts = cfb.replace(":", ";").split(";")
        if parts:
            try:
                bg = int(parts[-1])
                if 7 <= bg <= 15:
                    return "light"
                if bg <= 6:
                    return "dark"
            except Exception:
                pass
    colorterm = os.environ.get("COLORTERM", "").lower()
    if colorterm in ("truecolor", "24bit"):
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
    """Return colors for theme: bg, fg, chatbox_bg, input_fg."""
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
    else:
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
    lvl = (level or "").strip().lower()
    if lvl not in INTEL_CHOICES:
        raise ValueError(f"intelligence must be one of {', '.join(INTEL_CHOICES)}, got {level!r}")
    return lvl


@dataclass
class PickerItem:
    kind: str  # "provider" | "model"
    provider: str
    label: str  # display text (provider name or model id)
    is_provider_header: bool = False


def build_picker_items(provider_models: dict[str, list[str]]) -> list[PickerItem]:
    """Flatten {provider: [models]} into a linear list for navigation."""
    items: list[PickerItem] = []
    for prov in PROVIDER_NAMES:
        items.append(PickerItem(kind="provider", provider=prov, label=prov, is_provider_header=True))
        for m in provider_models.get(prov, []) or []:
            items.append(PickerItem(kind="model", provider=prov, label=m, is_provider_header=False))
    return items


def parse_slash_command(text: str) -> tuple[str, list[str]]:
    """Return (cmd, args) for slash input. E.g. '/intel high' -> ('intel', ['high'])"""
    t = text.strip()
    if not t.startswith("/"):
        return ("", [])
    parts = t[1:].split()
    if not parts:
        return ("", [])
    return (parts[0].lower(), parts[1:])


# ---------------------------------------------------------------------------
# Curses application — real interactive shell
# ---------------------------------------------------------------------------

class TuiApp:
    """Stateful curses app. Created per run_tui invocation."""

    def __init__(self, config: AgentConfig, client: Any | None = None) -> None:
        self.config = config
        self.client = client
        self.mode = config.mode.upper() if config.mode.upper() in MODE_ORDER else "AUTO"
        if self.mode == "SAFE":
            self.mode = "AUTO"
        self.provider = config.provider
        self.model = config.model
        self.intelligence = config.intelligence
        self.theme = config.theme
        self.input_text = ""
        self.cursor_pos = 0
        # Real conversation history — list of dicts {role, content}
        self.messages: list[dict[str, str]] = []
        # Compatibility: history as list[str] for tests that append strings
        self.history: list[str] = []
        self.status_msg = "TAB: switch mode  |  /models  /connect  /intel  /help  |  Ctrl+C cancel  /quit exit"
        self.should_quit = False
        # EventHub + TaskRunner for real backend
        self._hub = None
        self._runner = None
        self._scroll_offset = 0
        self._pending_status = ""
        # For streaming dedup
        self._last_event_count = 0

    # -- state mutators (also persist) -------------------------------------

    def cycle_mode(self) -> None:
        self.mode = next_mode(self.mode)
        try:
            save_tui_state({"mode": self.mode})
        except Exception:
            pass

    def set_intelligence(self, level: str) -> str:
        lvl = validate_intel(level)
        self.intelligence = lvl
        n_ctx, n_pred, c_budget, _ = intelligence_values(lvl)
        try:
            save_tui_state({
                "intelligence": lvl,
                "num_ctx": n_ctx,
                "num_predict": n_pred,
                "context_budget_chars": c_budget,
            })
        except Exception:
            pass
        return f"Intelligence → {lvl} ({n_ctx}/{n_pred}, budget {c_budget})"

    def set_provider_model(self, provider: str, model: str, base_url: str | None = None) -> None:
        self.provider = provider
        if model:
            self.model = model
        data: dict[str, Any] = {"provider": provider, "model": self.model}
        if base_url:
            data["ollama_base_url" if provider == "ollama" else f"{provider}_base_url"] = base_url
        try:
            save_tui_state(data)
        except Exception:
            pass

    def _live_config(self, workspace: Path | None = None) -> AgentConfig:
        """Build a live AgentConfig reflecting current UI state (mode/model/intel)."""
        overrides: dict[str, Any] = {
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "intelligence": self.intelligence,
            "theme": self.theme,
        }
        if workspace is not None:
            overrides["workspace"] = workspace
        else:
            # preserve original workspace
            overrides["workspace"] = self.config.workspace
        # intelligence_values will fill num_ctx etc. via load_config logic
        try:
            return load_config(**overrides)
        except Exception:
            return self.config

    def _add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.history.append(content if role == "user" else f"{role}: {content}")
        # auto-scroll to bottom
        self._scroll_offset = 0

    def _add_system(self, content: str) -> None:
        self._add_message("system", content)

    # -- lazy hub/runner init ----------------------------------------------

    def _ensure_hub_runner(self):
        if self._hub is None:
            try:
                from .web import EventHub, TaskRunner
                from .workspace import Workspace
            except Exception:
                return None, None
            self._hub = EventHub()
            # runner created per-task to capture live_config, so keep hub only
        return self._hub, None

    def _is_busy(self) -> bool:
        if self._runner is None:
            return False
        try:
            return bool(self._runner.busy)
        except Exception:
            return False

    def _start_task(self, text: str) -> bool:
        """Start a real AgentLoop task. Returns True if started, False if busy/error."""
        if self._is_busy():
            self._add_system("Already running — press Ctrl+C to cancel.")
            return False
        # Build live config
        try:
            live_cfg = self._live_config()
        except Exception as e:
            self._add_system(f"Config error: {e}")
            return False
        # Ensure hub
        if self._hub is None:
            from .web import EventHub
            self._hub = EventHub()
        # Resolve client
        client = self.client
        if client is None:
            try:
                from .ollama import OllamaClient
                client = OllamaClient(
                    base_url=live_cfg.ollama_base_url,
                    model=live_cfg.model,
                    request_timeout=live_cfg.request_timeout,
                    keep_alive=live_cfg.keep_alive,
                    num_ctx=live_cfg.num_ctx,
                    num_predict=live_cfg.num_predict,
                )
            except Exception as e:
                self._add_system(f"Ollama client error: {e}")
                return False
        # Workspace
        try:
            from .workspace import Workspace
            ws = Workspace(live_cfg.workspace)
        except Exception as e:
            self._add_system(f"Workspace error: {e}")
            return False
        # Create runner per-task so config is fresh
        try:
            from .web import TaskRunner
            self._runner = TaskRunner(live_cfg, client, ws, self._hub)
            ok = self._runner.start(text, mode=live_cfg.mode)
            if not ok:
                self._add_system("Task already running.")
                return False
            self._add_message("user", text)
            self.status_msg = f"Running ({live_cfg.mode}) — Ctrl+C to cancel"
            self._last_event_count = len(self._hub.history())
            return True
        except Exception as e:
            self._add_system(f"Failed to start task: {e}")
            return False

    def _poll_runner(self) -> None:
        """Drain hub events into messages and update status."""
        if self._hub is None:
            return
        try:
            hist = self._hub.history()
        except Exception:
            return
        # Process new events since last poll
        if len(hist) <= self._last_event_count:
            # also check if runner finished
            if self._runner is not None:
                try:
                    if not self._runner.busy and self._runner.result is not None:
                        res = self._runner.result
                        # Map LoopResult / GraphLoopResult to message
                        summary = getattr(res, "summary", "") or getattr(res, "error", "") or ""
                        status = getattr(res, "status", "")
                        if status in ("completed", "COMPLETE"):
                            self._add_message("assistant", summary or "Done.")
                            self.status_msg = "Completed — ready"
                        elif status in ("failed", "FAILED"):
                            self._add_system(f"Failed: {summary}")
                            self.status_msg = "Failed — ready"
                        elif status in ("cancelled", "CANCELLED"):
                            self._add_system("Cancelled.")
                            self.status_msg = "Cancelled — ready"
                        else:
                            if summary:
                                self._add_message("assistant", summary)
                        self._runner = None
                except Exception:
                    pass
            return
        new_events = hist[self._last_event_count:]
        self._last_event_count = len(hist)
        for ev in new_events:
            try:
                etype = getattr(ev, "type", "") or ev.get("type", "") if isinstance(ev, dict) else getattr(ev, "type", "")
                msg = getattr(ev, "message", "") or (ev.get("message", "") if isinstance(ev, dict) else "")
                # Map key events to system/assistant messages
                if etype in ("agent_started", "status"):
                    if msg:
                        self.status_msg = msg
                elif etype in ("thinking", "activity"):
                    if msg:
                        self.status_msg = msg
                elif etype == "model_started":
                    self.status_msg = "Thinking…"
                elif etype == "model_completed":
                    self.status_msg = "Processing…"
                elif etype == "tool_started":
                    tool = getattr(ev, "tool", "") or (ev.get("tool", "") if isinstance(ev, dict) else "")
                    if tool:
                        self._add_system(f"→ {tool}")
                elif etype == "tool_completed":
                    tool = getattr(ev, "tool", "") or (ev.get("tool", "") if isinstance(ev, dict) else "")
                    output = getattr(ev, "output", "") or getattr(ev, "message", "") or ""
                    if isinstance(ev, dict):
                        output = ev.get("output", "") or ev.get("message", "")
                    # Truncate for display
                    if output and len(output) > 800:
                        output = output[:800] + "…"
                    if tool and output:
                        # Only show concise tool completion
                        pass
                elif etype == "command_output":
                    out = getattr(ev, "output", "") or msg
                    if out:
                        if len(out) > 800:
                            out = out[:800] + "…"
                        self._add_system(out.strip())
                elif etype == "file_written":
                    target = getattr(ev, "target", "") or msg
                    if target:
                        self._add_system(f"Wrote {target}")
                elif etype == "patch_applied":
                    target = getattr(ev, "target", "") or msg
                    if target:
                        self._add_system(f"Patched {target}")
                elif etype in ("agent_completed", "task_completed"):
                    # Final summary will be handled via runner.result
                    if msg:
                        self._add_message("assistant", msg)
                elif etype == "agent_error":
                    if msg:
                        self._add_system(f"Error: {msg}")
                elif etype == "task_failed":
                    if msg:
                        self._add_system(f"Task failed: {msg}")
                elif etype in ("task_plan",):
                    if msg:
                        self._add_system(msg)
            except Exception:
                continue
        # Check completion after draining
        if self._runner is not None:
            try:
                if not self._runner.busy and self._runner.result is not None:
                    res = self._runner.result
                    summary = getattr(res, "summary", "") or ""
                    err = getattr(res, "error", "") or ""
                    status = getattr(res, "status", "")
                    if err and status not in ("completed", "COMPLETE"):
                        self._add_system(f"{status}: {err or summary}")
                        self.status_msg = f"{status} — ready"
                    elif summary and not any(m["content"] == summary for m in self.messages[-2:]):
                        # Avoid duplicate if agent_completed already added
                        self._add_message("assistant", summary)
                        self.status_msg = "Completed — ready"
                    else:
                        if not self._runner.busy:
                            self.status_msg = "Ready — TAB: switch mode  |  /models  /connect  /intel  /help"
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
            self._add_system("Cancelled by user (Ctrl+C).")
        except Exception as e:
            self.status_msg = f"Cancel failed: {e}"

    # -- curses drawing ----------------------------------------------------

    def _init_colors(self, stdscr) -> None:
        if not HAS_CURSES or curses is None:
            return
        try:
            curses.use_default_colors()
            curses.curs_set(1)
        except Exception:
            pass
        try:
            if curses.has_colors():
                curses.start_color()
                tc = theme_colors(self.theme)
                use_256 = getattr(curses, "COLORS", 8) >= 256
                try:
                    if use_256:
                        bg = tc["bg_idx"]
                        cbg = tc["chatbox_bg_idx"]
                        fg = tc["fg_idx"]
                        curses.init_pair(1, fg, bg)
                        curses.init_pair(8, fg, cbg)
                        curses.init_pair(9, fg, bg)
                        curses.init_pair(2, MODE_COLOR_IDX["PLAN"], bg)
                        curses.init_pair(3, MODE_COLOR_IDX["BUILD"], bg)
                        curses.init_pair(4, MODE_COLOR_IDX["AUTO"], bg)
                        curses.init_pair(10, MODE_COLOR_IDX["PLAN"], cbg)
                        curses.init_pair(11, MODE_COLOR_IDX["BUILD"], cbg)
                        curses.init_pair(12, MODE_COLOR_IDX["AUTO"], cbg)
                        curses.init_pair(5, PINK_FG_IDX, PINK_BG_IDX)
                        curses.init_pair(6, curses.COLOR_WHITE, PINK_BG_IDX)
                        curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_CYAN)
                    else:
                        bg8 = curses.COLOR_BLACK if tc["theme"] == "dark" else curses.COLOR_WHITE
                        fg8 = curses.COLOR_WHITE if tc["theme"] == "dark" else curses.COLOR_BLACK
                        curses.init_pair(1, fg8, bg8)
                        curses.init_pair(8, fg8, bg8)
                        curses.init_pair(9, fg8, bg8)
                        curses.init_pair(2, curses.COLOR_YELLOW, bg8)
                        curses.init_pair(3, curses.COLOR_BLUE, bg8)
                        curses.init_pair(4, curses.COLOR_RED, bg8)
                        curses.init_pair(10, curses.COLOR_YELLOW, bg8)
                        curses.init_pair(11, curses.COLOR_BLUE, bg8)
                        curses.init_pair(12, curses.COLOR_RED, bg8)
                        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_MAGENTA)
                except Exception:
                    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
                    curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_BLACK)
                    curses.init_pair(9, curses.COLOR_WHITE, curses.COLOR_BLACK)
                    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
                    curses.init_pair(3, curses.COLOR_BLUE, curses.COLOR_BLACK)
                    curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
                    curses.init_pair(10, curses.COLOR_YELLOW, curses.COLOR_BLACK)
                    curses.init_pair(11, curses.COLOR_BLUE, curses.COLOR_BLACK)
                    curses.init_pair(12, curses.COLOR_RED, curses.COLOR_BLACK)
                    curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_MAGENTA)
        except Exception:
            pass

    def _wrap_lines(self, text: str, width: int) -> list[str]:
        """Word-wrap text to width."""
        if width <= 0:
            return [text]
        lines: list[str] = []
        for para in text.split("\n"):
            if not para:
                lines.append("")
                continue
            while len(para) > width:
                # try to break at space
                cut = para.rfind(" ", 0, width)
                if cut <= width // 2:
                    cut = width
                lines.append(para[:cut])
                para = para[cut:].lstrip()
            lines.append(para)
        return lines

    def _draw(self, stdscr) -> None:
        if not HAS_CURSES or curses is None:
            return
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        tc = theme_colors(self.theme)
        geom = calc_chatbox_geometry(h, w)
        tier = geom.get("tier", "normal")  # type: ignore
        try:
            if curses.has_colors():
                try:
                    stdscr.bkgd(" ", curses.color_pair(9))
                except Exception:
                    stdscr.bkgd(" ", curses.color_pair(1))
                stdscr.erase()
        except Exception:
            pass

        if geom.get("too_small"):  # type: ignore
            try:
                msg1 = "Terminal too small."
                msg2 = "Resize the terminal to continue."
                stdscr.addstr(max(0, h // 2 - 1), max(0, (w - len(msg1)) // 2), msg1, curses.A_BOLD if curses else 0)
                stdscr.addstr(max(0, h // 2), max(0, (w - len(msg2)) // 2), msg2, curses.A_DIM if curses else 0)
                mode = self.mode
                col = {"PLAN": 2, "BUILD": 3, "AUTO": 4}.get(mode, 1)
                try:
                    if h >= 2 and w >= 10:
                        stdscr.addstr(h - 1, 1, f"[{mode}]", curses.color_pair(col) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD)
                        footer = format_model_footer(self.model, self.intelligence)
                        if len(footer) < w - 4:
                            stdscr.addstr(h - 1, w - len(footer) - 1, footer, curses.color_pair(1) if curses.has_colors() else 0)
                except curses.error:
                    pass
            except curses.error:
                pass
            stdscr.refresh()
            return

        if geom["is_minimised"]:
            try:
                msg = " — minimised — resize larger " if tier == "minimised" else " — compact — "
                stdscr.addstr(0, max(0, (w - len(msg)) // 2), msg, curses.A_BOLD if curses else 0)
                mode = self.mode
                col = {"PLAN": 2, "BUILD": 3, "AUTO": 4}.get(mode, 1)
                stdscr.addstr(h - 2, 1, f"[{mode}]", curses.color_pair(col) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD)
                footer = format_model_footer(self.model, self.intelligence)
                if len(footer) < w - 2:
                    stdscr.addstr(h - 2, w - len(footer) - 1, footer, curses.color_pair(1) if curses.has_colors() else 0)
                prompt = "> " + self.input_text
                stdscr.addstr(h - 1, 0, prompt[: w - 1])
            except curses.error:
                pass
            stdscr.refresh()
            return

        chat_h = geom["chat_h"]
        chat_w = geom["chat_w"]
        chat_y = geom["chat_y"]
        chat_x = geom["chat_x"]
        inner_w = geom["inner_w"]
        inner_h = geom["inner_h"]

        # Chatbox window
        try:
            chat_win = curses.newwin(chat_h, chat_w, chat_y, chat_x)
            try:
                if curses.has_colors():
                    chat_win.bkgd(" ", curses.color_pair(8))
            except Exception:
                pass
            chat_win.box()
            title = " A.S.C.S "
            try:
                chat_win.addstr(0, max(1, (chat_w - len(title)) // 2), title, curses.A_BOLD)
            except curses.error:
                pass

            # Render real conversation
            # Build wrapped lines with role prefixes
            rendered: list[tuple[str, int]] = []  # (text, attr)
            if not self.messages:
                # Welcome / empty state — not fake queue, just guidance
                welcome = "Welcome to A.S.C.S — type a request or /help"
                wrapped = self._wrap_lines(welcome, max(10, inner_w - 2))
                for ln in wrapped:
                    rendered.append((ln, curses.A_DIM if curses else 0))
            else:
                for m in self.messages:
                    role = m.get("role", "")
                    content = m.get("content", "")
                    prefix = ""
                    attr = 0
                    if role == "user":
                        prefix = "You: "
                        attr = curses.A_BOLD if curses else 0
                    elif role == "assistant":
                        prefix = "ASCS: "
                        attr = 0
                    elif role == "system":
                        prefix = "· "
                        attr = curses.A_DIM if curses else 0
                    # Wrap content with prefix
                    full = prefix + content
                    # For system, keep dim
                    wrapped = self._wrap_lines(full, max(10, inner_w - 2))
                    for idx, ln in enumerate(wrapped):
                        # indent continuation
                        if idx > 0 and prefix:
                            ln = "  " + ln
                        rendered.append((ln, attr))

            # Apply scroll_offset (0 = bottom)
            # Show last inner_h lines
            if len(rendered) > inner_h:
                start = max(0, len(rendered) - inner_h - self._scroll_offset)
                end = start + inner_h
                visible = rendered[start:end]
            else:
                visible = rendered

            for idx, (ln, attr) in enumerate(visible):
                y = 1 + idx
                if y >= chat_h - 1:
                    break
                # Truncate to inner width
                if len(ln) > inner_w:
                    ln = ln[: max(0, inner_w - 1)] + "…"
                try:
                    # Use chatbox pair 8 for all content to keep bg consistent
                    pair = curses.color_pair(8) if curses.has_colors() else 0
                    chat_win.addstr(y, 2, ln[:inner_w], pair | attr)
                except curses.error:
                    pass

            # Bottom line inside chatbox: mode left, model(intel) right (both on chatbox bg)
            footer = format_model_footer(self.model, self.intelligence)
            mode_str = f" {self.mode} "
            col_chat = {"PLAN": 10, "BUILD": 11, "AUTO": 12}.get(self.mode, 8)
            try:
                chat_win.addstr(chat_h - 2, 2, mode_str, curses.color_pair(col_chat) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD)
                if tier == "compact":
                    max_footer = inner_w - 8
                    if len(footer) > max_footer:
                        footer = footer[: max_footer - 1] + "…"
                if len(footer) + 4 < inner_w:
                    chat_win.addstr(chat_h - 2, chat_w - len(footer) - 3, footer, curses.color_pair(8) if curses.has_colors() else 0)
                else:
                    short = footer[: max(0, inner_w - 6)] + "…"
                    chat_win.addstr(chat_h - 2, chat_w - len(short) - 3, short, curses.color_pair(8) if curses.has_colors() else 0)
            except curses.error:
                pass
            chat_win.noutrefresh()
        except curses.error:
            pass

        # Input window — on interface bg
        try:
            inp_h = 3
            inp_y = chat_y + chat_h + 1
            if inp_y + inp_h > h:
                inp_y = h - inp_h
            inp_w = chat_w
            inp_x = chat_x
            inp_win = curses.newwin(inp_h, inp_w, inp_y, inp_x)
            try:
                if curses.has_colors():
                    inp_win.bkgd(" ", curses.color_pair(9))
            except Exception:
                pass
            inp_win.box()
            prompt = "> "
            max_input = inner_w - len(prompt) - 1 if inner_w > 10 else w - 10
            if max_input < 5:
                max_input = 5
            total_len = len(self.input_text)
            if total_len <= max_input:
                visible = self.input_text
                cursor_col = len(prompt) + self.cursor_pos
            else:
                if self.cursor_pos <= max_input:
                    visible = self.input_text[:max_input]
                    cursor_col = len(prompt) + self.cursor_pos
                elif self.cursor_pos >= total_len:
                    visible = self.input_text[-max_input:]
                    cursor_col = len(prompt) + len(visible)
                else:
                    half = max_input // 2
                    start = max(0, self.cursor_pos - half)
                    end = min(total_len, start + max_input)
                    if end - start < max_input:
                        start = max(0, end - max_input)
                    visible = self.input_text[start:end]
                    cursor_col = len(prompt) + (self.cursor_pos - start)
                    cursor_col = max(len(prompt), min(cursor_col, len(prompt) + len(visible)))
            tc_pair = curses.color_pair(9) if curses.has_colors() else 0
            inp_win.addstr(1, 1, prompt, curses.A_BOLD | tc_pair)
            inp_win.addstr(1, 1 + len(prompt), visible, tc_pair | curses.A_BOLD)
            try:
                inp_win.move(1, min(inp_w - 2, cursor_col + 1))
                inp_win.noutrefresh()
                stdscr.move(inp_y + 1, inp_x + min(inp_w - 2, cursor_col + 1))
            except curses.error:
                inp_win.noutrefresh()
        except curses.error:
            pass

        # Status bar
        try:
            # Show busy indicator if running
            bar = self.status_msg
            if self._is_busy():
                bar = "● " + bar
            stdscr.addstr(h - 1, 0, bar[: w - 1].ljust(w - 1)[: w - 1], curses.A_DIM)
        except curses.error:
            pass
        stdscr.noutrefresh()
        try:
            curses.doupdate()
        except Exception:
            try:
                stdscr.refresh()
            except Exception:
                pass

    def _handle_input_key(self, ch: int, stdscr) -> bool:
        """Return True if input was submitted (Enter)."""
        if ch in (10, 13, curses.KEY_ENTER if HAS_CURSES and curses else 10):
            return True
        if ch in (8, 127, curses.KEY_BACKSPACE if HAS_CURSES and curses else 127, 263):
            if self.cursor_pos > 0:
                self.input_text = self.input_text[: self.cursor_pos - 1] + self.input_text[self.cursor_pos :]
                self.cursor_pos -= 1
            elif self.input_text:
                self.input_text = self.input_text[:-1]
                self.cursor_pos = len(self.input_text)
            return False
        if ch == 330 or (HAS_CURSES and ch == curses.KEY_DC):
            if 0 <= self.cursor_pos < len(self.input_text):
                self.input_text = self.input_text[: self.cursor_pos] + self.input_text[self.cursor_pos + 1 :]
            return False
        if ch == 260 or (HAS_CURSES and ch == curses.KEY_LEFT):
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
            return False
        if ch == 261 or (HAS_CURSES and ch == curses.KEY_RIGHT):
            if self.cursor_pos < len(self.input_text):
                self.cursor_pos += 1
            return False
        # Home / End
        if ch == 262 or (HAS_CURSES and ch == curses.KEY_HOME):
            self.cursor_pos = 0
            return False
        if ch == 360 or (HAS_CURSES and ch == curses.KEY_END):
            self.cursor_pos = len(self.input_text)
            return False
        # Ctrl+A (1) = Home, Ctrl+E (5) = End
        if ch == 1:
            self.cursor_pos = 0
            return False
        if ch == 5:
            self.cursor_pos = len(self.input_text)
            return False
        if 32 <= ch <= 126:
            c = chr(ch)
            self.input_text = self.input_text[: self.cursor_pos] + c + self.input_text[self.cursor_pos :]
            self.cursor_pos += 1
            return False
        return False

    def _run_picker(self, stdscr, provider_models: dict[str, list[str]]) -> tuple[str, str] | None:
        if not HAS_CURSES or curses is None:
            return None
        items = build_picker_items(provider_models)
        sel = 0
        for idx, it in enumerate(items):
            if it.is_provider_header and it.provider == self.provider:
                sel = idx
                break
        h, w = stdscr.getmaxyx()
        picker_h = min(len(items) + 4, h - 4 if h > 4 else 20)
        picker_w_raw = min(60, w - 4 if w > 4 else 60)
        title = " Select provider / model — Enter to confirm, Esc to cancel "
        min_w = min(len(title) + 4, w - 2 if w > 2 else len(title) + 4)
        picker_w = max(min_w, picker_w_raw)
        picker_w = min(picker_w, w - 2 if w > 2 else picker_w)
        if picker_h < 5:
            picker_h = 5
        if picker_w < 20:
            picker_w = min(60, w - 2 if w > 2 else 60)
        picker_y = (h - picker_h) // 2
        picker_x = (w - picker_w) // 2
        while True:
            h, w = stdscr.getmaxyx()
            picker_h = min(len(items) + 4, h - 4 if h > 4 else 20)
            picker_w_raw = min(60, w - 4 if w > 4 else 60)
            picker_w = max(min_w, picker_w_raw)
            picker_w = min(picker_w, w - 2 if w > 2 else picker_w)
            if picker_h < 5:
                picker_h = 5
            picker_y = max(0, (h - picker_h) // 2)
            picker_x = max(0, (w - picker_w) // 2)
            try:
                win = curses.newwin(picker_h, picker_w, picker_y, picker_x)
                win.box()
                win.addstr(0, max(1, (picker_w - len(title)) // 2), title[: picker_w - 2], curses.A_BOLD)
                visible_start = max(0, sel - (picker_h - 4) // 2)
                visible_end = min(len(items), visible_start + picker_h - 3)
                if visible_end - visible_start < picker_h - 3 and visible_start > 0:
                    visible_start = max(0, visible_end - (picker_h - 3))
                for i in range(visible_start, visible_end):
                    it = items[i]
                    y = 1 + i - visible_start
                    if it.is_provider_header:
                        txt = f" {it.provider} "
                        attr = curses.A_BOLD
                    else:
                        txt = f"   {it.label}"
                        attr = 0
                    txt = txt[: max(0, picker_w - 2)].ljust(max(0, picker_w - 2))
                    if i == sel:
                        try:
                            hl = curses.color_pair(5) | curses.A_BOLD if curses.has_colors() else curses.A_REVERSE
                            win.addstr(y, 1, txt, hl)
                        except curses.error:
                            win.addstr(y, 1, txt, curses.A_REVERSE)
                    else:
                        if it.is_provider_header:
                            win.addstr(y, 1, txt, curses.A_BOLD)
                        else:
                            win.addstr(y, 1, txt, attr)
                win.noutrefresh()
                curses.doupdate()
            except curses.error:
                pass
            try:
                ch = stdscr.getch()
            except Exception:
                return None
            if ch in (27,):
                return None
            if ch in (10, 13, curses.KEY_ENTER if hasattr(curses, "KEY_ENTER") else 10):
                chosen = items[sel]
                if chosen.is_provider_header:
                    return (chosen.provider, "")
                else:
                    return (chosen.provider, chosen.label)
            if ch in (curses.KEY_UP if HAS_CURSES else 259, 259):
                sel = max(0, sel - 1)
            elif ch in (curses.KEY_DOWN if HAS_CURSES else 258, 258):
                sel = min(len(items) - 1, sel + 1)
            elif ch == 9:
                sel = min(len(items) - 1, sel + 1)
            elif ch == curses.KEY_RESIZE if HAS_CURSES and hasattr(curses, "KEY_RESIZE") else 410:
                pass

    def _run_intel_picker(self, stdscr) -> str | None:
        if not HAS_CURSES or curses is None:
            return None
        h, w = stdscr.getmaxyx()
        picker_h = len(INTEL_DISPLAY_ORDER) + 4
        picker_w = 36
        picker_h = min(picker_h, h - 4 if h > 4 else picker_h)
        picker_w = min(picker_w, w - 4 if w > 4 else picker_w)
        if picker_h < 5:
            picker_h = 5
        picker_y = (h - picker_h) // 2
        picker_x = (w - picker_w) // 2
        sel = 0
        for idx, lvl in enumerate(INTEL_DISPLAY_ORDER):
            if lvl == self.intelligence:
                sel = idx
                break
        title = " Intelligence — Enter to confirm, Esc to cancel "
        while True:
            h, w = stdscr.getmaxyx()
            picker_y = max(0, (h - picker_h) // 2)
            picker_x = max(0, (w - picker_w) // 2)
            try:
                win = curses.newwin(picker_h, picker_w, picker_y, picker_x)
                win.box()
                win.addstr(0, max(1, (picker_w - len(title)) // 2), title[: picker_w - 2], curses.A_BOLD)
                for i, lvl in enumerate(INTEL_DISPLAY_ORDER):
                    y = 1 + i
                    if y >= picker_h - 1:
                        break
                    n_ctx, n_pred, _, _ = intelligence_values(lvl)
                    if lvl == self.intelligence:
                        label = f"★ {lvl} ({n_ctx}/{n_pred}) "
                    else:
                        label = f"  {lvl} ({n_ctx}/{n_pred}) "
                    txt = label[: max(0, picker_w - 2)].ljust(max(0, picker_w - 2))
                    if i == sel:
                        try:
                            hl = curses.color_pair(5) | curses.A_BOLD if curses.has_colors() else curses.A_REVERSE
                            win.addstr(y, 1, txt, hl)
                        except curses.error:
                            win.addstr(y, 1, txt, curses.A_REVERSE)
                    else:
                        win.addstr(y, 1, txt, 0)
                win.addstr(picker_h - 1, 2, "↑/↓ move  Enter select  Esc cancel"[: picker_w - 4], curses.A_DIM)
                win.noutrefresh()
                curses.doupdate()
            except curses.error:
                pass
            try:
                ch = stdscr.getch()
            except Exception:
                return None
            if ch in (27,):
                return None
            if ch in (10, 13, curses.KEY_ENTER if hasattr(curses, "KEY_ENTER") else 10):
                return INTEL_DISPLAY_ORDER[sel]
            if ch in (curses.KEY_UP if HAS_CURSES else 259, 259):
                sel = max(0, sel - 1)
            elif ch in (curses.KEY_DOWN if HAS_CURSES else 258, 258):
                sel = min(len(INTEL_DISPLAY_ORDER) - 1, sel + 1)
            elif ch == 9:
                sel = min(len(INTEL_DISPLAY_ORDER) - 1, sel + 1)
            elif ch == curses.KEY_RESIZE if HAS_CURSES and hasattr(curses, "KEY_RESIZE") else 410:
                pass

    def _do_connect(self, stdscr) -> None:
        from .providers import list_all_providers_with_models
        h, w = stdscr.getmaxyx()
        self.status_msg = "Fetching provider models…"
        self._draw(stdscr)
        try:
            provider_models = list_all_providers_with_models(timeout=2, use_cache=True)
        except Exception:
            provider_models = {p: [] for p in PROVIDER_NAMES}
        self.status_msg = "TAB: switch mode  |  /models  /connect  /intel  /help"
        res = self._run_picker(stdscr, provider_models)
        if not res:
            return
        prov, _ = res
        if not HAS_CURSES or curses is None:
            return
        h, w = stdscr.getmaxyx()
        dialog_h = 9
        dialog_w = min(64, w - 4)
        dialog_y = (h - dialog_h) // 2
        dialog_x = (w - dialog_w) // 2
        from .providers import DEFAULT_BASE_URLS
        cur_base = DEFAULT_BASE_URLS.get(prov, "")
        persisted = load_tui_state()
        per_prov = persisted.get("providers", {}).get(prov, {}) if isinstance(persisted.get("providers"), dict) else {}
        if isinstance(per_prov, dict) and per_prov.get("base_url"):
            cur_base = str(per_prov["base_url"])
        base_url = cur_base
        api_key = ""
        from .providers import API_KEY_ENVS
        env_key = API_KEY_ENVS.get(prov)
        if env_key and os.environ.get(env_key):
            api_key = os.environ.get(env_key, "")
        field = 0
        buf = [base_url, api_key]
        labels = ["Base URL:", "API Key (leave empty for none):"]
        fields = 1 if prov == "ollama" else 2
        while True:
            h, w = stdscr.getmaxyx()
            dialog_y = max(0, (h - dialog_h) // 2)
            dialog_x = max(0, (w - dialog_w) // 2)
            dialog_w = min(64, w - 4 if w > 4 else 64)
            if dialog_w < 20:
                dialog_w = w - 2 if w > 2 else 20
            try:
                win = curses.newwin(dialog_h, dialog_w, dialog_y, dialog_x)
                win.box()
                title = f" Connect — {prov} "
                win.addstr(0, max(0, (dialog_w - len(title)) // 2), title, curses.A_BOLD)
                for idx in range(fields):
                    y = 2 + idx * 2
                    if y + 1 >= dialog_h - 1:
                        continue
                    win.addstr(y, 2, labels[idx][: max(0, dialog_w - 4)])
                    txt = buf[idx][: max(0, dialog_w - 4)]
                    if idx == 1 and txt:
                        disp = "*" * len(txt)
                    else:
                        disp = txt
                    attr = curses.color_pair(5) if idx == field and HAS_CURSES and curses.has_colors() else (curses.A_REVERSE if idx == field else 0)
                    win.addstr(y + 1, 2, disp.ljust(max(0, dialog_w - 4))[: max(0, dialog_w - 4)], attr)
                win.addstr(dialog_h - 2, 2, "Enter: confirm  Tab: next  Esc: cancel"[: max(0, dialog_w - 4)], curses.A_DIM)
                win.noutrefresh()
                curses.doupdate()
            except curses.error:
                pass
            try:
                ch = stdscr.getch()
            except Exception:
                return
            if ch == 27:
                return
            if ch == 9:
                field = (field + 1) % fields
                continue
            if ch in (10, 13):
                new_base = buf[0].strip() or cur_base
                new_key = buf[1].strip() if fields > 1 else ""
                from .providers import list_models_for_provider
                models = list_models_for_provider(prov, base_url=new_base, api_key=new_key or None, timeout=5, use_cache=False)
                try:
                    state = load_tui_state()
                    provs = state.get("providers", {})
                    if not isinstance(provs, dict):
                        provs = {}
                    provs[prov] = {"base_url": new_base}
                    if new_key and prov != "ollama":
                        provs[prov]["api_key"] = new_key
                    state["providers"] = provs
                    state["provider"] = prov
                    save_tui_state(state)
                    if env_key and new_key:
                        os.environ[env_key] = new_key
                    self.provider = prov
                    self.status_msg = f"Connected to {prov} ({len(models)} models)" if models else f"Connected to {prov} (no models listed)"
                    self._add_system(self.status_msg)
                except Exception as e:
                    self.status_msg = f"Connect failed: {e}"
                    self._add_system(self.status_msg)
                return
            if ch in (curses.KEY_BACKSPACE, 127, 8, 263) if HAS_CURSES else (127, 8):
                if buf[field]:
                    buf[field] = buf[field][:-1]
            elif 32 <= ch <= 126:
                buf[field] += chr(ch)
            elif ch == curses.KEY_RESIZE if HAS_CURSES and hasattr(curses, "KEY_RESIZE") else 410:
                continue

    def _handle_slash(self, stdscr, text: str) -> None:
        cmd, args = parse_slash_command(text)
        if not cmd:
            self._add_system("Unknown command — try /help")
            return
        if cmd in ("help", "?"):
            help_text = (
                "Commands: /models  /connect  /intel [low|medium|high|xhigh|default]  /status  /clear  /history  /experiences  /tasks  /check  /quit\n"
                "Keys: TAB=mode  Enter=send  Esc=quit  Ctrl+C=cancel  Home/End, Arrows, Backspace/Delete  Up/Down scroll"
            )
            self._add_system(help_text)
            self.status_msg = "Help shown"
            return
        if cmd in ("clear", "cls"):
            self.messages.clear()
            self.history.clear()
            if self._hub is not None:
                try:
                    self._hub.clear()
                except Exception:
                    pass
            self._scroll_offset = 0
            self.status_msg = "Cleared"
            self._add_system("Cleared")
            # keep the system message; clear again would remove it, so keep 1
            return
        if cmd in ("quit", "exit", "q"):
            self.should_quit = True
            return
        if cmd == "intel":
            if not args:
                if HAS_CURSES and curses is not None and stdscr is not None:
                    chosen = self._run_intel_picker(stdscr)
                    if chosen:
                        try:
                            msg = self.set_intelligence(chosen)
                            self.status_msg = msg
                            self._add_system(msg)
                        except ValueError as e:
                            self.status_msg = str(e)
                            self._add_system(str(e))
                    else:
                        self.status_msg = f"Intelligence cancelled (current: {self.intelligence})"
                    return
                self._add_system(f"Usage: /intel {'|'.join(INTEL_CHOICES)}  (current: {self.intelligence})")
                return
            try:
                msg = self.set_intelligence(args[0])
                self.status_msg = msg
                self._add_system(msg)
            except ValueError as e:
                self.status_msg = str(e)
                self._add_system(str(e))
            return
        if cmd == "models":
            from .providers import list_all_providers_with_models
            try:
                provider_models = list_all_providers_with_models(timeout=2, use_cache=True)
            except Exception:
                provider_models = {p: [] for p in PROVIDER_NAMES}
                self._add_system("Unable to load models.")
            res = self._run_picker(stdscr, provider_models)
            if res:
                prov, model = res
                if model:
                    self.set_provider_model(prov, model)
                    self.status_msg = f"Model → {prov}/{model}"
                    self._add_system(self.status_msg)
                else:
                    self.set_provider_model(prov, "")
                    self.status_msg = f"Provider → {prov} (no model)"
                    self._add_system(self.status_msg)
            else:
                self.status_msg = "Model selection cancelled"
            return
        if cmd == "connect":
            try:
                self._do_connect(stdscr)
            except Exception:
                self.status_msg = "Unable to connect provider."
                self._add_system(self.status_msg)
            return
        if cmd == "status":
            # Real state
            try:
                live = self._live_config()
                ws = str(live.workspace)
            except Exception:
                ws = str(self.config.workspace)
            conn = "unknown"
            try:
                from .ollama import OllamaClient
                c = self.client or OllamaClient(model=self.model)
                rep = c.ensure_ready(check_timeout=5, prewarm=False)
                conn = "ready" if rep.get("available") else f"model {self.model!r} not installed"
                if not rep.get("reachable"):
                    conn = "Ollama unreachable"
            except Exception as e:
                conn = f"error: {e}"
            status = (
                f"Provider: {self.provider}\n"
                f"Model: {self.model}\n"
                f"Intelligence: {self.intelligence}\n"
                f"Mode: {self.mode}\n"
                f"Workspace: {ws}\n"
                f"Connection: {conn}\n"
                f"Theme: {self.theme} ({detect_theme(self.theme)})\n"
                f"Busy: {self._is_busy()}"
            )
            self._add_system(status)
            self.status_msg = "Status shown"
            return
        if cmd == "history":
            if not self.messages:
                self._add_system("No history yet.")
            else:
                hist = "\n".join(f"{m['role']}: {m['content']}" for m in self.messages[-20:])
                self._add_system(hist or "No history")
            return
        if cmd == "experiences":
            try:
                from .experience import ExperienceStore
                store = ExperienceStore()
                exps = store.recent(limit=5)
                if not exps:
                    self._add_system("No experiences yet.")
                else:
                    out = "\n".join(f"- {e.task[:80]} → {'success' if e.success else 'failed'} (score {e.score:.2f})" for e in exps)
                    self._add_system(out)
            except Exception as e:
                self._add_system(f"Experiences error: {e}")
            return
        if cmd == "tasks":
            try:
                from .project import ProjectStore
                store = ProjectStore(self._live_config().workspace)
                graph = store.load_task_graph()
                if not graph:
                    self._add_system("No saved tasks (.ascs/task_state.json empty).")
                else:
                    txt = f"Tasks: {len(graph.tasks)} — " + ", ".join(f"{t.id}:{t.status}" for t in list(graph.tasks.values())[:10])
                    self._add_system(txt)
            except Exception as e:
                self._add_system(f"Tasks error: {e}")
            return
        if cmd == "check":
            try:
                from .doctor import doctor
                rep = doctor(workspace=str(self._live_config().workspace))
                lines = []
                for r in rep.results:
                    lines.append(f"{r.status} {r.name}: {r.message}")
                self._add_system("\n".join(lines))
            except Exception as e:
                self._add_system(f"Check failed: {e}")
            return
        if cmd == "clear":
            self.messages.clear()
            self._add_system("Cleared")
            return
        self._add_system(f"Unknown command /{cmd} — try /help")

    def run_curses(self, stdscr) -> None:
        self._init_colors(stdscr)
        stdscr.keypad(True)
        try:
            curses.cbreak()
        except Exception:
            pass
        stdscr.timeout(100)
        try:
            curses.curs_set(1)
        except Exception:
            pass
        # Welcome message (not fake queue)
        if not self.messages:
            self._add_system(f"A.S.C.S ready — {self.model}({self.intelligence}) on {self.provider} — /help for commands")
        self._draw(stdscr)
        while not self.should_quit:
            # Poll runner before input so streaming shows immediately
            self._poll_runner()
            # Redraw periodically even without input for streaming
            self._draw(stdscr)
            try:
                ch = stdscr.get_wch()  # type: ignore[attr-defined]
                if isinstance(ch, str):
                    if ch == "\t":
                        self.cycle_mode()
                        self._draw(stdscr)
                        continue
                    if ch == "\n" or ch == "\r":
                        text = self.input_text.strip()
                        if text:
                            self.input_text = ""
                            self.cursor_pos = 0
                            if text.startswith("/"):
                                self._handle_slash(stdscr, text)
                            else:
                                self._start_task(text)
                        self._draw(stdscr)
                        continue
                    if ch == "\x1b":  # ESC
                        if self._is_busy():
                            self._cancel_running()
                        else:
                            self.should_quit = True
                            break
                    if ch == "\x03":  # Ctrl+C
                        if self._is_busy():
                            self._cancel_running()
                        else:
                            self.input_text = ""
                            self.cursor_pos = 0
                            self.status_msg = "Cleared input — Ctrl+C again to quit, /quit to exit"
                        self._draw(stdscr)
                        continue
                    if ch == "\x04":  # Ctrl+D
                        self.should_quit = True
                        break
                    if ch == "\x7f" or ch == "\b":
                        if self.cursor_pos > 0:
                            self.input_text = self.input_text[: self.cursor_pos - 1] + self.input_text[self.cursor_pos :]
                            self.cursor_pos -= 1
                        elif self.input_text:
                            self.input_text = self.input_text[:-1]
                            self.cursor_pos = len(self.input_text)
                        self._draw(stdscr)
                        continue
                    if len(ch) == 1 and 32 <= ord(ch) <= 126 or ord(ch) > 127:
                        self.input_text = self.input_text[: self.cursor_pos] + ch + self.input_text[self.cursor_pos :]
                        self.cursor_pos += 1
                        self._draw(stdscr)
                        continue
                else:
                    if ch == 9:  # TAB
                        self.cycle_mode()
                        self._draw(stdscr)
                        continue
                    if ch == curses.KEY_RESIZE if HAS_CURSES else 410:
                        self._draw(stdscr)
                        continue
                    if ch in (curses.KEY_ENTER if HAS_CURSES else 10, 10, 13):
                        text = self.input_text.strip()
                        if text:
                            self.input_text = ""
                            self.cursor_pos = 0
                            if text.startswith("/"):
                                self._handle_slash(stdscr, text)
                            else:
                                self._start_task(text)
                        self._draw(stdscr)
                        continue
                    if ch == 3:  # Ctrl+C as int
                        if self._is_busy():
                            self._cancel_running()
                        else:
                            self.input_text = ""
                            self.cursor_pos = 0
                        self._draw(stdscr)
                        continue
                    if ch == 4:  # Ctrl+D
                        self.should_quit = True
                        break
                    if ch == curses.KEY_UP if HAS_CURSES else 259:
                        # scroll up
                        if self._scroll_offset < max(0, len(self.messages) * 2):
                            self._scroll_offset += 1
                        self._draw(stdscr)
                        continue
                    if ch == curses.KEY_DOWN if HAS_CURSES else 258:
                        if self._scroll_offset > 0:
                            self._scroll_offset -= 1
                        self._draw(stdscr)
                        continue
                    # Delegate to input handler for arrows/home/end etc.
                    if ch in (260, 261, 262, 360, 330, 263, 127, 8):
                        self._handle_input_key(ch, stdscr)
                        self._draw(stdscr)
                        continue
                    # Home/End via get_wch may be single char, handle via fallback
                    self._handle_input_key(ch, stdscr)
                    self._draw(stdscr)
                    continue
            except curses.error:
                continue
            except KeyboardInterrupt:
                if self._is_busy():
                    self._cancel_running()
                else:
                    self.should_quit = True
                    break
        # Cleanup: cancel running task
        if self._is_busy():
            try:
                self._cancel_running()
            except Exception:
                pass

    def run_fallback(self) -> int:
        """Non-interactive fallback — minimal, no fake preview."""
        print("A.S.C.S. TUI requires a real terminal.", file=sys.stderr)
        print("Run from PowerShell/Terminal:  .\\risa.cmd --tui  or  python -m agent --tui", file=sys.stderr)
        if not HAS_CURSES:
            print("curses not available — on Windows: pip install windows-curses", file=sys.stderr)
        return 1


def run_tui(config: AgentConfig, client: Any | None = None, *, block: bool = True) -> int:
    """Entry point for `risa --tui` — real full-screen TUI."""
    app = TuiApp(config, client)
    # Real TTY check — fail fast, no fake line-mode preview
    if not HAS_CURSES or curses is None:
        print("curses not available on this platform.", file=sys.stderr)
        print("On Windows, install with: pip install windows-curses", file=sys.stderr)
        print("Then run: .\\risa.cmd --tui", file=sys.stderr)
        return 1
    if not sys.stdout.isatty() or not sys.stdin.isatty():
        print("TUI requires a real terminal (TTY).", file=sys.stderr)
        print("You are in a non-interactive shell (piped input / OpenCode tool).", file=sys.stderr)
        print("Open a real terminal and run: python -m agent --tui", file=sys.stderr)
        return 1
    term = os.environ.get("TERM", "")
    if not term or term == "dumb":
        # Don't fail hard, just warn — many Windows terminals have no TERM
        os.environ["TERM"] = "xterm-256color"
    try:
        return curses.wrapper(app.run_curses)  # type: ignore
    except curses.error as e:
        print(f"Curses error: {e}", file=sys.stderr)
        print("Ensure TERM=xterm-256color and terminal size >= 40x10", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        # Ensure terminal restored — curses.wrapper does endwin, but be explicit
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
