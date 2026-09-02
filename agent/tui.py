"""Interactive TUI for A.S.C.S.

Features (spec-driven):
  - TAB cycles Plan(orange) -> Build(blue) -> Auto(red)
  - /models  -> provider-aware model picker (bold provider, pink highlight)
  - /connect -> provider connector (local + cloud)
  - /intel   -> low/medium/high/xhigh/default  -> (num_ctx, num_predict, budget, level)
  - Chatbox rectangle sized for: "Hello, hello, hello, hello, hello, hello, hello, hello, hello"
  - Theme-aware (auto/light/dark) black/white bg, contrast input, slightly offset chatbox
  - Minimised layout when terminal shrinks
  - Persistence across restarts via tui_state.json

Output area is placeholder per spec (deferred).

Zero extra deps: stdlib curses (optional on Windows via windows-curses).
"""

from __future__ import annotations

import os
import sys
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
    load_tui_state,
    save_tui_state,
    tui_state_path,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HELLO_TEXT = "Hello, hello, hello, hello, hello, hello, hello, hello, hello"
# Minimum inner width must fit HELLO_TEXT exactly (63 chars) + 2 padding on each side for nice border spacing.
# Spec: big enough to show exactly this text => inner >= len(HELLO_TEXT)
HELLO_LEN = len(HELLO_TEXT)  # 63
MIN_CHATBOX_INNER_W = HELLO_LEN  # 63
MIN_CHATBOX_W = MIN_CHATBOX_INNER_W + 2  # 65 inc borders
MIN_CHATBOX_H = 5  # at least 3 content lines + 2 borders
MIN_TERM_W = 40
MIN_TERM_H = 10

MODE_ORDER = ("PLAN", "BUILD", "AUTO")
MODE_COLORS = {
    "PLAN": "orange",
    "BUILD": "blue",
    "AUTO": "red",
}
# ANSI 256 indices for mode colors
MODE_COLOR_IDX = {"PLAN": 208, "BUILD": 27, "AUTO": 196}
PINK_BG_IDX = 213  # pink highlight (approx #FF87D7) — any pink in 200-219 range satisfies spec
PINK_FG_IDX = 16  # black text on pink for contrast, alternative white handled in drawing

INTEL_CHOICES = ("low", "medium", "high", "xhigh", "default")

# Try to import curses optionally.
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


def calc_chatbox_geometry(term_h: int, term_w: int) -> dict[str, int]:
    """Return geometry for chatbox given terminal size.

    Returns dict with keys: chat_h, chat_w, chat_y, chat_x, is_minimised, inner_w, inner_h
    """
    is_min = term_h < 12 or term_w < 50
    if is_min:
        # minimised: chatbox hidden or 3 lines only
        return {
            "chat_h": 0,
            "chat_w": 0,
            "chat_y": 0,
            "chat_x": 0,
            "is_minimised": 1,
            "inner_w": 0,
            "inner_h": 0,
        }
    # Normal: chatbox occupies most of screen minus input/footer
    # input area is 3 rows + header 1
    avail_h = term_h - 5  # 1 header + 3 input + 1 footer margin
    avail_w = term_w - 2  # side margins
    chat_h = max(MIN_CHATBOX_H, min(avail_h, term_h - 5))
    chat_w = max(MIN_CHATBOX_W, avail_w)
    # Center chatbox if extra width
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
    }


def is_minimised(term_h: int, term_w: int) -> bool:
    return bool(calc_chatbox_geometry(term_h, term_w)["is_minimised"])


def detect_theme(config_theme: str) -> str:
    """Resolve 'auto' to 'light' or 'dark' based on env hints.

    Uses COLORFGBG, TERM, and OS dark-mode hints. Falls back to 'dark'.
    """
    t = (config_theme or "auto").lower()
    if t in ("light", "dark"):
        return t
    # auto detection
    # COLORFGBG is "fg;bg" where bg 0-6 dark, 7-8 light on many terms
    cfb = os.environ.get("COLORFGBG", "")
    if cfb:
        parts = cfb.split(";")
        if parts:
            try:
                bg = int(parts[-1])
                if bg >= 7 and bg <= 15:
                    return "light"
                if bg <= 6:
                    return "dark"
            except Exception:
                pass
    term = os.environ.get("TERM", "").lower()
    if "light" in term:
        return "light"
    # macOS dark mode? not exposed; default dark
    return "dark"


def theme_colors(theme: str) -> dict[str, Any]:
    """Return colors for theme: bg, fg, chatbox_bg, input_fg.

    Spec (re-checked):
      - Interface BG: black (dark) or white (light) per device theme
      - Input text: contrast (white on black, black on white)
      - Chatbox BG: grey (variant: dark grey for dark, light grey for light) — both grey family
      - 1-char padding at edges, but background fills entire tab
    """
    resolved = detect_theme(theme)
    if resolved == "light":
        return {
            "theme": "light",
            "bg": "white",
            "bg_idx": 15,  # white
            "fg": "black",
            "fg_idx": 0,
            "chatbox_bg": "grey_light",
            "chatbox_bg_idx": 250,  # #bcbcbc medium grey — clearly grey, darker than white
            "input_fg": "black",
            "input_fg_idx": 0,
            "border_fg": "black",
            "border_fg_idx": 0,
        }
    else:
        return {
            "theme": "dark",
            "bg": "black",
            "bg_idx": 16,  # black (0 also black, 16 is consistent 256)
            "fg": "white",
            "fg_idx": 15,
            "chatbox_bg": "grey_dark",
            "chatbox_bg_idx": 235,  # #262626 slightly lighter than black
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


# ---------------------------------------------------------------------------
# Provider picker model (testable)
# ---------------------------------------------------------------------------

@dataclass
class PickerItem:
    kind: str  # "provider" | "model"
    provider: str
    label: str  # display text (provider name or model id)
    is_provider_header: bool = False


def build_picker_items(provider_models: dict[str, list[str]]) -> list[PickerItem]:
    """Flatten {provider: [models]} into a linear list for navigation.

    Provider header is always present (bold). Models follow indented.
    Empty model list => only header row (still selectable, per spec).
    """
    items: list[PickerItem] = []
    for prov in PROVIDER_NAMES:
        items.append(PickerItem(kind="provider", provider=prov, label=prov, is_provider_header=True))
        for m in provider_models.get(prov, []) or []:
            items.append(PickerItem(kind="model", provider=prov, label=m, is_provider_header=False))
    return items


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

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
# Curses application
# ---------------------------------------------------------------------------

class TuiApp:
    """Stateful curses app. Created per run_tui invocation."""

    def __init__(self, config: AgentConfig, client: Any | None = None) -> None:
        self.config = config
        self.client = client
        # Mutable runtime state (mirrors config but allows TAB/intel changes)
        self.mode = config.mode.upper() if config.mode.upper() in MODE_ORDER else "AUTO"
        if self.mode == "SAFE":
            self.mode = "AUTO"
        self.provider = config.provider
        self.model = config.model
        self.intelligence = config.intelligence
        self.theme = config.theme
        self.input_text = ""
        self.cursor_pos = 0
        self.history: list[str] = []
        self.status_msg = "TAB: switch mode  |  /models  /connect  /intel  /help"
        self.should_quit = False

    # -- state mutators (also persist) -------------------------------------

    def cycle_mode(self) -> None:
        self.mode = next_mode(self.mode)
        # persist mode? we store in tui_state so restart keeps it
        try:
            save_tui_state({"mode": self.mode})
        except Exception:
            pass

    def set_intelligence(self, level: str) -> str:
        lvl = validate_intel(level)
        self.intelligence = lvl
        n_ctx, n_pred, c_budget, _ = intelligence_values(lvl)
        # update config fields for next runs (in-memory copy)
        # config is frozen, so we keep local values and persist
        try:
            save_tui_state({
                "intelligence": lvl,
                "num_ctx": n_ctx,
                "num_predict": n_pred,
                "context_budget_chars": c_budget,
            })
        except Exception:
            pass
        # also update in-memory config view for footer
        # we mutate the dataclass via object.__setattr__ for frozen, but safer to just keep separate
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
                # Detect 256-color support
                use_256 = getattr(curses, "COLORS", 8) >= 256
                try:
                    if use_256:
                        # Interface and chatbox backgrounds per spec:
                        # dark: 16 black + 235 lighter; light: 15 white + 254 darker
                        bg = tc["bg_idx"]
                        cbg = tc["chatbox_bg_idx"]
                        fg = tc["fg_idx"]
                        # Pair 1: input text (contrast) on interface bg
                        curses.init_pair(1, fg, bg)
                        # Pair 8: chatbox content (same fg, offset bg)
                        curses.init_pair(8, fg, cbg)
                        # Pair 9: interface bg itself (for stdscr)
                        curses.init_pair(9, fg, bg)
                        # Mode colors on interface bg (orange/blue/red)
                        curses.init_pair(2, MODE_COLOR_IDX["PLAN"], bg)
                        curses.init_pair(3, MODE_COLOR_IDX["BUILD"], bg)
                        curses.init_pair(4, MODE_COLOR_IDX["AUTO"], bg)
                        # Mode on chatbox bg (for mode/footer inside chatbox)
                        curses.init_pair(10, MODE_COLOR_IDX["PLAN"], cbg)
                        curses.init_pair(11, MODE_COLOR_IDX["BUILD"], cbg)
                        curses.init_pair(12, MODE_COLOR_IDX["AUTO"], cbg)
                        # Pink highlight: black on pink (full row)
                        curses.init_pair(5, PINK_FG_IDX, PINK_BG_IDX)
                        curses.init_pair(6, curses.COLOR_WHITE, PINK_BG_IDX)
                        curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_CYAN)
                    else:
                        # 8-color fallback: use solid black/white, chatbox offset via dim
                        bg8 = curses.COLOR_BLACK if tc["theme"] == "dark" else curses.COLOR_WHITE
                        fg8 = curses.COLOR_WHITE if tc["theme"] == "dark" else curses.COLOR_BLACK
                        curses.init_pair(1, fg8, bg8)
                        curses.init_pair(8, fg8, bg8)
                        curses.init_pair(9, fg8, bg8)
                        curses.init_pair(2, curses.COLOR_YELLOW, bg8)  # orange approx
                        curses.init_pair(3, curses.COLOR_BLUE, bg8)
                        curses.init_pair(4, curses.COLOR_RED, bg8)
                        curses.init_pair(10, curses.COLOR_YELLOW, bg8)
                        curses.init_pair(11, curses.COLOR_BLUE, bg8)
                        curses.init_pair(12, curses.COLOR_RED, bg8)
                        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_MAGENTA)
                except Exception:
                    # ultimate fallback 8 colors
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

    def _draw(self, stdscr) -> None:
        if not HAS_CURSES or curses is None:
            return
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        tc = theme_colors(self.theme)
        geom = calc_chatbox_geometry(h, w)

        # Interface background per spec: solid black (dark) or white (light)
        # Chatbox uses offset bg (8), stdscr uses interface bg (9)
        try:
            if curses.has_colors():
                # Prefer pair 9 (interface), fall back to 1
                try:
                    stdscr.bkgd(" ", curses.color_pair(9))
                except Exception:
                    stdscr.bkgd(" ", curses.color_pair(1))
                # Also erase with that bg
                stdscr.erase()
        except Exception:
            pass

        # If minimised, show compact view
        if geom["is_minimised"]:
            try:
                msg = " — minimised — resize larger "
                stdscr.addstr(0, max(0, (w - len(msg)) // 2), msg, curses.A_BOLD if curses else 0)
                # show mode + footer even minimised
                mode = self.mode
                col = {"PLAN": 2, "BUILD": 3, "AUTO": 4}.get(mode, 1)
                stdscr.addstr(h - 2, 1, f"[{mode}]", curses.color_pair(col) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD)
                footer = format_model_footer(self.model, self.intelligence)
                if len(footer) < w - 2:
                    stdscr.addstr(h - 2, w - len(footer) - 1, footer, curses.color_pair(1) if curses.has_colors() else 0)
                # input line
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

        # Chatbox window — offset bg per spec (lighter for dark, darker for light)
        try:
            chat_win = curses.newwin(chat_h, chat_w, chat_y, chat_x)
            try:
                if curses.has_colors():
                    chat_win.bkgd(" ", curses.color_pair(8))
            except Exception:
                pass
            chat_win.box()
            # Title centered
            title = " A.S.C.S — chat "
            try:
                chat_win.addstr(0, max(1, (chat_w - len(title)) // 2), title, curses.A_BOLD)
            except curses.error:
                pass
            # Show hello placeholder centered vertically
            # Ensure inner area can fit hello text exactly
            display = HELLO_TEXT
            if len(display) > inner_w:
                display = display[: max(0, inner_w - 1)] + "…" if inner_w > 1 else ""
            y_mid = inner_h // 2
            x_mid = max(1, (inner_w - len(display)) // 2)
            try:
                # Hello centered on chatbox bg (same fg as input, but chatbox offset bg)
                chat_win.addstr(1 + y_mid, 1 + x_mid, display, curses.color_pair(8) if curses.has_colors() else 0)
            except curses.error:
                pass
            # Bottom line inside chatbox: mode left, model(intel) right (both on chatbox bg)
            footer = format_model_footer(self.model, self.intelligence)
            mode_str = f" {self.mode} "
            # Mode on chatbox bg uses pairs 10-12 so bg matches chatbox
            col_chat = {"PLAN": 10, "BUILD": 11, "AUTO": 12}.get(self.mode, 8)
            try:
                chat_win.addstr(chat_h - 2, 2, mode_str, curses.color_pair(col_chat) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD)
                # model footer same colour as input but on chatbox bg (pair 8)
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

        # Input window — on interface bg (not chatbox), contrast per spec
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
            visible = self.input_text[-max_input:] if len(self.input_text) > max_input else self.input_text
            tc_pair = curses.color_pair(9) if curses.has_colors() else 0
            inp_win.addstr(1, 1, prompt, curses.A_BOLD | tc_pair)
            inp_win.addstr(1, 1 + len(prompt), visible, tc_pair | curses.A_BOLD)
            # status line below input or above footer
            inp_win.noutrefresh()
        except curses.error:
            pass

        # Status bar at very bottom of screen
        try:
            stdscr.addstr(h - 1, 0, self.status_msg[: w - 1].ljust(w - 1)[: w - 1], curses.A_DIM)
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
        # Enter
        if ch in (10, 13, curses.KEY_ENTER if HAS_CURSES and curses else 10):
            return True
        # Backspace
        if ch in (8, 127, curses.KEY_BACKSPACE if HAS_CURSES and curses else 127, 263):
            if self.cursor_pos > 0:
                self.input_text = self.input_text[: self.cursor_pos - 1] + self.input_text[self.cursor_pos :]
                self.cursor_pos -= 1
            elif self.input_text:
                self.input_text = self.input_text[:-1]
                self.cursor_pos = len(self.input_text)
            return False
        # Delete
        if HAS_CURSES and ch == curses.KEY_DC:
            if 0 <= self.cursor_pos < len(self.input_text):
                self.input_text = self.input_text[: self.cursor_pos] + self.input_text[self.cursor_pos + 1 :]
            return False
        # Left/Right
        if HAS_CURSES and ch == curses.KEY_LEFT:
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
            return False
        if HAS_CURSES and ch == curses.KEY_RIGHT:
            if self.cursor_pos < len(self.input_text):
                self.cursor_pos += 1
            return False
        # Printable
        if 32 <= ch <= 126:
            c = chr(ch)
            self.input_text = self.input_text[: self.cursor_pos] + c + self.input_text[self.cursor_pos :]
            self.cursor_pos += 1
            return False
        # Unicode via get_wch alternative handled in loop
        return False

    def _run_picker(self, stdscr, provider_models: dict[str, list[str]]) -> tuple[str, str] | None:
        """Show provider/model picker overlay. Returns (provider, model) or None."""
        if not HAS_CURSES or curses is None:
            return None
        items = build_picker_items(provider_models)
        # Start selected at current provider header
        sel = 0
        for idx, it in enumerate(items):
            if it.is_provider_header and it.provider == self.provider:
                sel = idx
                break
        picker_h = min(len(items) + 4, curses.LINES - 4 if hasattr(curses, "LINES") else 20)
        picker_w = min(60, curses.COLS - 4 if hasattr(curses, "COLS") else 60)
        # Fallback to stdscr size
        h, w = stdscr.getmaxyx()
        picker_h = min(picker_h, h - 4)
        picker_w = min(picker_w, w - 4)
        picker_y = (h - picker_h) // 2
        picker_x = (w - picker_w) // 2
        # Keys: up/down, enter, esc
        while True:
            h, w = stdscr.getmaxyx()
            picker_y = max(0, (h - picker_h) // 2)
            picker_x = max(0, (w - picker_w) // 2)
            try:
                win = curses.newwin(picker_h, picker_w, picker_y, picker_x)
                win.box()
                title = " Select provider / model — Enter to confirm, Esc to cancel "
                win.addstr(0, max(1, (picker_w - len(title)) // 2), title, curses.A_BOLD)
                # Help line
                # Visible window
                visible_start = max(0, sel - (picker_h - 4) // 2)
                visible_end = min(len(items), visible_start + picker_h - 3)
                # Adjust if at end
                if visible_end - visible_start < picker_h - 3 and visible_start > 0:
                    visible_start = max(0, visible_end - (picker_h - 3))
                for i in range(visible_start, visible_end):
                    it = items[i]
                    y = 1 + i - visible_start
                    # Prepare line: provider bold, model indented
                    if it.is_provider_header:
                        txt = f" {it.provider} "
                        attr = curses.A_BOLD
                    else:
                        txt = f"   {it.label}"
                        attr = 0
                    # Truncate to fit
                    txt = txt[: picker_w - 2].ljust(picker_w - 2)
                    # Pink highlight on selected entire row
                    if i == sel:
                        # full-row pink bg
                        try:
                            # selected model text colour turns (to black on pink vs white)
                            # we use pair 5 for selected
                            hl = curses.color_pair(5) | curses.A_BOLD if curses.has_colors() else curses.A_REVERSE
                            win.addstr(y, 1, txt, hl)
                        except curses.error:
                            win.addstr(y, 1, txt, curses.A_REVERSE)
                    else:
                        # provider bold even when not selected
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
            if ch in (27,):  # ESC
                return None
            if ch in (10, 13, curses.KEY_ENTER if hasattr(curses, "KEY_ENTER") else 10):
                chosen = items[sel]
                if chosen.is_provider_header:
                    # Selecting provider header with no model: return provider with empty model => keeps current model but switches provider
                    return (chosen.provider, "")
                else:
                    return (chosen.provider, chosen.label)
            if ch in (curses.KEY_UP if HAS_CURSES else 259, 259):
                sel = max(0, sel - 1)
            elif ch in (cuses_KEY_DOWN := (curses.KEY_DOWN if HAS_CURSES else 258), 258):
                sel = min(len(items) - 1, sel + 1)
            elif ch == 9:  # TAB in picker also moves down
                sel = min(len(items) - 1, sel + 1)
            elif ch == curses.KEY_RESIZE if HAS_CURSES and hasattr(curses, "KEY_RESIZE") else 410:
                # just redraw
                pass

    def _do_connect(self, stdscr) -> None:
        """Interactive /connect flow: pick provider then prompt for base_url/api_key."""
        # Reuse picker but also allow provider_models to be fetched first for display
        from .providers import list_all_providers_with_models

        # Fetch models for all providers; show status but don't block long.
        # Parallel fetch with 2s per-provider keeps this <2s total (was 5s sequential).
        h, w = stdscr.getmaxyx()
        self.status_msg = "Fetching provider models…"
        self._draw(stdscr)
        try:
            provider_models = list_all_providers_with_models(timeout=2, use_cache=True)
        except Exception:
            provider_models = {p: [] for p in PROVIDER_NAMES}
        self.status_msg = "TAB: switch mode  |  /models  /connect  /intel  /help"
        # Show picker
        res = self._run_picker(stdscr, provider_models)
        if not res:
            return
        prov, _ = res
        # Now prompt for base_url / api_key in a small dialog
        # Use simple curses input windows
        if not HAS_CURSES or curses is None:
            return
        h, w = stdscr.getmaxyx()
        dialog_h = 9
        dialog_w = min(64, w - 4)
        dialog_y = (h - dialog_h) // 2
        dialog_x = (w - dialog_w) // 2
        # Get existing values
        from .providers import DEFAULT_BASE_URLS

        cur_base = DEFAULT_BASE_URLS.get(prov, "")
        # Try to load persisted per-provider base url
        persisted = load_tui_state()
        per_prov = persisted.get("providers", {}).get(prov, {}) if isinstance(persisted.get("providers"), dict) else {}
        if isinstance(per_prov, dict) and per_prov.get("base_url"):
            cur_base = str(per_prov["base_url"])

        # Simple two-field prompt: base_url then api_key
        # For ollama, only base_url is needed
        base_url = cur_base
        api_key = ""
        # For cloud, we need key; prefill from env if present
        from .providers import API_KEY_ENVS

        env_key = API_KEY_ENVS.get(prov)
        if env_key and os.environ.get(env_key):
            api_key = os.environ.get(env_key, "")

        # Dialog loop
        field = 0  # 0=base_url, 1=api_key
        buf = [base_url, api_key]
        labels = ["Base URL:", "API Key (leave empty for none):"]
        # Don't show API key field for ollama
        fields = 1 if prov == "ollama" else 2
        while True:
            try:
                win = curses.newwin(dialog_h, dialog_w, dialog_y, dialog_x)
                win.box()
                title = f" Connect — {prov} "
                win.addstr(0, (dialog_w - len(title)) // 2, title, curses.A_BOLD)
                for idx in range(fields):
                    y = 2 + idx * 2
                    win.addstr(y, 2, labels[idx][: dialog_w - 4])
                    # input field highlight: pink bg if selected
                    txt = buf[idx][: dialog_w - 4]
                    # mask api key
                    if idx == 1 and txt:
                        disp = "*" * len(txt)
                    else:
                        disp = txt
                    attr = curses.color_pair(5) if idx == field and HAS_CURSES and curses.has_colors() else (curses.A_REVERSE if idx == field else 0)
                    win.addstr(y + 1, 2, disp.ljust(dialog_w - 4)[: dialog_w - 4], attr)
                win.addstr(dialog_h - 2, 2, "Enter: confirm  Tab: next  Esc: cancel".ljust(dialog_w - 4)[: dialog_w - 4], curses.A_DIM)
                win.noutrefresh()
                curses.doupdate()
            except curses.error:
                pass
            try:
                ch = stdscr.getch()
            except Exception:
                return
            if ch == 27:  # ESC
                return
            if ch == 9:  # TAB
                field = (field + 1) % fields
                continue
            if ch in (10, 13):
                # confirm
                new_base = buf[0].strip() or cur_base
                new_key = buf[1].strip() if fields > 1 else ""
                # Validate by listing
                from .providers import list_models_for_provider

                models = list_models_for_provider(prov, base_url=new_base, api_key=new_key or None, timeout=5, use_cache=False)
                # Persist
                try:
                    state = load_tui_state()
                    provs = state.get("providers", {})
                    if not isinstance(provs, dict):
                        provs = {}
                    provs[prov] = {"base_url": new_base}
                    # Don't store raw key in state file in plain? But spec says persist across restarts
                    # We store key if provided, else keep existing env handling. For ollama no key.
                    if new_key and prov != "ollama":
                        provs[prov]["api_key"] = new_key  # persisted, file 600
                    state["providers"] = provs
                    state["provider"] = prov
                    if models:
                        # keep model if existing belongs to provider
                        pass
                    save_tui_state(state)
                    # Also update env for current session so list_models uses it
                    if env_key and new_key:
                        os.environ[env_key] = new_key
                    self.provider = prov
                    self.status_msg = f"Connected to {prov} ({len(models)} models)" if models else f"Connected to {prov} (no models listed)"
                except Exception as e:
                    self.status_msg = f"Connect failed: {e}"
                return
            # typing
            if ch in (curses.KEY_BACKSPACE, 127, 8, 263) if HAS_CURSES else (127, 8):
                if buf[field]:
                    buf[field] = buf[field][:-1]
            elif 32 <= ch <= 126:
                buf[field] += chr(ch)
            elif ch == curses.KEY_RESIZE if HAS_CURSES and hasattr(curses, "KEY_RESIZE") else 410:
                h, w = stdscr.getmaxyx()
                dialog_y = (h - dialog_h) // 2
                dialog_x = (w - dialog_w) // 2

    def _handle_slash(self, stdscr, text: str) -> None:
        cmd, args = parse_slash_command(text)
        if not cmd:
            self.status_msg = "Unknown command"
            return
        if cmd in ("help", "?"):
            self.status_msg = "Commands: /models  /connect  /intel [low|medium|high|xhigh|default]  /clear  /quit  TAB=mode"
            return
        if cmd in ("clear", "cls"):
            self.history.clear()
            self.status_msg = "Cleared"
            return
        if cmd in ("quit", "exit", "q"):
            self.should_quit = True
            return
        if cmd == "intel":
            if not args:
                self.status_msg = f"Usage: /intel {'|'.join(INTEL_CHOICES)}  (current: {self.intelligence})"
                return
            try:
                msg = self.set_intelligence(args[0])
                self.status_msg = msg
            except ValueError as e:
                self.status_msg = str(e)
            return
        if cmd == "models":
            # Show picker; models fetched per provider (parallel, 2s max)
            from .providers import list_all_providers_with_models

            try:
                provider_models = list_all_providers_with_models(timeout=2, use_cache=True)
            except Exception:
                provider_models = {p: [] for p in PROVIDER_NAMES}
            res = self._run_picker(stdscr, provider_models)
            if res:
                prov, model = res
                if model:
                    self.set_provider_model(prov, model)
                    self.status_msg = f"Model → {prov}/{model}"
                else:
                    # provider selected but no model -> just switch provider
                    self.set_provider_model(prov, "")
                    self.status_msg = f"Provider → {prov} (no model)"
            else:
                self.status_msg = "Model selection cancelled"
            return
        if cmd == "connect":
            self._do_connect(stdscr)
            return
        self.status_msg = f"Unknown command /{cmd} — try /help"

    def run_curses(self, stdscr) -> None:
        self._init_colors(stdscr)
        stdscr.keypad(True)
        # Enable mouse? not needed
        try:
            curses.cbreak()
        except Exception:
            pass
        # Non-blocking? Use blocking getch with timeout for resize
        stdscr.timeout(100)  # 100ms poll for resize
        nodelay = False
        try:
            curses.curs_set(1)
        except Exception:
            pass
        self._draw(stdscr)
        pending_resize = False
        while not self.should_quit:
            try:
                ch = stdscr.get_wch()  # type: ignore[attr-defined]
                # get_wch returns str for unicode, int for special
                if isinstance(ch, str):
                    if ch == "\t":
                        self.cycle_mode()
                        self._draw(stdscr)
                        continue
                    if ch == "\n" or ch == "\r":
                        text = self.input_text.strip()
                        if text:
                            self.history.append(text)
                            self.input_text = ""
                            self.cursor_pos = 0
                            if text.startswith("/"):
                                self._handle_slash(stdscr, text)
                            else:
                                # Placeholder for agent execution (output area deferred)
                                self.status_msg = f"({self.mode}) queued: {text[:40]}"
                        self._draw(stdscr)
                        continue
                    if ch == "\x1b":  # ESC
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
                    # Regular char
                    if len(ch) == 1 and 32 <= ord(ch) <= 126 or ord(ch) > 127:
                        self.input_text = self.input_text[: self.cursor_pos] + ch + self.input_text[self.cursor_pos :]
                        self.cursor_pos += 1
                        self._draw(stdscr)
                        continue
                    # ignore other
                else:
                    # int
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
                            self.history.append(text)
                            self.input_text = ""
                            self.cursor_pos = 0
                            if text.startswith("/"):
                                self._handle_slash(stdscr, text)
                            else:
                                self.status_msg = f"({self.mode}) queued: {text[:40]}"
                        self._draw(stdscr)
                        continue
                    if ch in (curses.KEY_BACKSPACE if HAS_CURSES else 263, 127, 8, 263):
                        if self.cursor_pos > 0:
                            self.input_text = self.input_text[: self.cursor_pos - 1] + self.input_text[self.cursor_pos :]
                            self.cursor_pos -= 1
                        elif self.input_text:
                            self.input_text = self.input_text[:-1]
                            self.cursor_pos = len(self.input_text)
                        self._draw(stdscr)
                        continue
                    if ch == curses.KEY_DC if HAS_CURSES else 330:
                        if 0 <= self.cursor_pos < len(self.input_text):
                            self.input_text = self.input_text[: self.cursor_pos] + self.input_text[self.cursor_pos + 1 :]
                            self._draw(stdscr)
                        continue
                    if ch == curses.KEY_LEFT if HAS_CURSES else 260:
                        if self.cursor_pos > 0:
                            self.cursor_pos -= 1
                            self._draw(stdscr)
                        continue
                    if ch == curses.KEY_RIGHT if HAS_CURSES else 261:
                        if self.cursor_pos < len(self.input_text):
                            self.cursor_pos += 1
                            self._draw(stdscr)
                        continue
                    # ignore other specials
            except curses.error:
                # timeout
                if pending_resize:
                    self._draw(stdscr)
                    pending_resize = False
                continue
            except Exception:
                # get_wch may raise on no input
                try:
                    stdscr.getch()
                except Exception:
                    pass
                continue

    def _preview_layout(self) -> None:
        """Render an ASCII preview of the full curses layout for headless inspection."""
        # Use 67 cols outer (61 inner) as minimal width that fits HELLO_TEXT, but expand to 78 for nicer preview
        w = 78
        inner = w - 2
        # Ensure HELLO_TEXT fits
        disp = HELLO_TEXT
        if len(disp) > inner - 4:
            disp = disp[: inner - 5] + "…"
        pad_left = (inner - len(disp)) // 2
        # Colors description
        tc = theme_colors(self.theme)
        print("")
        print("─" * w)
        print(f" Preview — full curses layout (Theme: {tc['theme']}, BG: {tc['bg']}, Input: {tc['input_fg']}) ")
        print("─" * w)
        # Chatbox
        title = " A.S.C.S — chat "
        print("┌" + title.center(w - 2, "─") + "┐")
        # 5 content lines, hello in middle
        for i in range(4):
            if i == 1:
                line = " " * pad_left + disp
                print("│" + line.ljust(inner) + "│")
            else:
                print("│" + " " * inner + "│")
        # Footer inside chatbox: mode left, model(intel) right
        footer = format_model_footer(self.model, self.intelligence)
        mode_disp = f"[{self.mode}]"
        # Show mode color hint
        gap = inner - len(mode_disp) - len(footer) - 4
        if gap < 2:
            gap = 2
        print("│  " + mode_disp + " " * gap + footer + "  │")
        print("└" + "─" * (w - 2) + "┘")
        # Input area
        print("┌" + " Input ".center(w - 2, "─") + "┐")
        print("│ > " + "_" * (inner - 4) + " │")
        print("└" + "─" * (w - 2) + "┘")
        print("TAB: switch mode  |  /models  /connect  /intel  /help".center(w))
        print(f"Colors: PLAN orange(208) BUILD blue(27) AUTO red(196)  •  Pink highlight 213 on selection".center(w))
        print(f"Minimise: resize <50×12 → compact (inner {MIN_CHATBOX_INNER_W} collapses) ".center(w))
        print("─" * w)
        print("")

    def run_fallback(self) -> int:
        """Line-mode fallback when curses unavailable."""
        print("A.S.C.S. TUI — fallback line mode (curses not available)")
        print(f"Mode: {self.mode} (TAB cycles PLAN->BUILD->AUTO)")
        print(f"Model: {format_model_footer(self.model, self.intelligence)}  Theme: {self.theme}")
        print(f"Chatbox inner width {MIN_CHATBOX_INNER_W} fits: {HELLO_TEXT!r}")
        print("Commands: /models  /connect  /intel [low|medium|high|xhigh|default]  /quit")
        # Show preview so user sees how full UI looks even without a TTY
        self._preview_layout()
        while not self.should_quit:
            try:
                line = input(f"[{self.mode}] {self.model}({self.intelligence})> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not line:
                continue
            if line == "\t":
                self.cycle_mode()
                print(f"Mode -> {self.mode}")
                continue
            if line.startswith("/"):
                cmd, args = parse_slash_command(line)
                if cmd == "intel":
                    if not args:
                        print(f"Usage: /intel {'|'.join(INTEL_CHOICES)} (current {self.intelligence})")
                    else:
                        try:
                            msg = self.set_intelligence(args[0])
                            print(msg)
                        except Exception as e:
                            print(e)
                elif cmd == "models":
                    print("[picker requires curses; listing providers]")
                    from .providers import list_all_providers_with_models

                    try:
                        pm = list_all_providers_with_models(timeout=2, use_cache=True)
                        for p in PROVIDER_NAMES:
                            print(f"  {p}: {pm.get(p, [])}")
                    except Exception as e:
                        print(f"error: {e}")
                elif cmd == "connect":
                    print("Use /connect in curses mode; here set PROVIDER env manually")
                elif cmd in ("quit", "exit", "q"):
                    break
                elif cmd in ("help", "?"):
                    print("Commands: /models /connect /intel /help /quit ; TAB cycles mode")
                else:
                    print(f"Unknown /{cmd}")
                continue
            # TAB simulation via literal \t input? also support "tab" command
            if line.lower() == "tab":
                self.cycle_mode()
                print(f"Mode -> {self.mode}")
                continue
            # Placeholder for agent run
            print(f"[{self.mode}] queued: {line[:80]} (output area deferred)")
        return 0


def run_tui(config: AgentConfig, client: Any | None = None, *, block: bool = True) -> int:
    """Entry point for `risa --tui`. Returns exit code."""
    app = TuiApp(config, client)
    if not HAS_CURSES or curses is None:
        print("curses not available on this platform; using line-mode fallback.", file=sys.stderr)
        print("On Windows, install with: pip install windows-curses", file=sys.stderr)
        return app.run_fallback()
    # OpenCode / CI / piped input is not a TTY — curses would fail with nocbreak().
    # Detect early and explain, then fall back cleanly instead of flashing escape codes.
    if not sys.stdout.isatty() or not sys.stdin.isatty():
        print("TUI requires a real terminal (TTY).", file=sys.stderr)
        print("You are in a non-interactive shell (OpenCode tool / piped input).", file=sys.stderr)
        print("→ Open macOS Terminal.app (or Windows Terminal) and run:", file=sys.stderr)
        print("  .venv/bin/python -m agent --tui   (Mac)  or  risa --tui", file=sys.stderr)
        print("Falling back to line-mode for this session…", file=sys.stderr)
        print("", file=sys.stderr)
        return app.run_fallback()
    # TERM check — curses needs a valid terminal type
    term = os.environ.get("TERM", "")
    if not term or term == "dumb":
        print(f"TERM={term!r} is not suitable for curses; set TERM=xterm-256color and retry.", file=sys.stderr)
        print("Falling back to line-mode…", file=sys.stderr)
        return app.run_fallback()
    try:
        return curses.wrapper(app.run_curses)  # type: ignore[arg-type]
    except curses.error as e:
        print(f"Curses unavailable ({e}), falling back to line mode.", file=sys.stderr)
        print("Hints: ensure TERM=xterm-256color and run in Terminal.app, not inside an IDE tool.", file=sys.stderr)
        return app.run_fallback()
    except KeyboardInterrupt:
        return 130
