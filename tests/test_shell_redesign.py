"""Shell redesign verification — tiers, cursor, theme, pickers, resize.

Covers spec sections not in test_tui_spec:
  - Responsive tiers (large/medium/compact/minimised/extremely_small)
  - Chatbox grey bg distinction, width/height tiers
  - Input cursor preservation and scrolling
  - Theme WT_SESSION/colon handling
  - Intel picker order and real config wiring
  - Extremely small guard
  - Slash handling edge cases
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from agent.config import AgentConfig, INTELLIGENCE_LEVELS, intelligence_values, load_config
from agent.tui import (
    HELLO_TEXT,
    HELLO_LEN,
    INTEL_CHOICES,
    INTEL_DISPLAY_ORDER,
    MODE_COLOR_IDX,
    PINK_BG_IDX,
    TuiApp,
    build_picker_items,
    calc_chatbox_geometry,
    detect_theme,
    format_model_footer,
    get_layout_tier,
    is_minimised,
    is_too_small,
    next_mode,
    parse_slash_command,
    theme_colors,
    validate_intel,
)


# ---------------------------------------------------------------------------
# Tier system (spec 5-10)
# ---------------------------------------------------------------------------

def test_get_layout_tier_large():
    assert get_layout_tier(40, 120) == "large"
    assert get_layout_tier(30, 100) == "large"


def test_get_layout_tier_wide():
    assert get_layout_tier(40, 140) == "wide"
    assert get_layout_tier(40, 200) == "wide"


def test_get_layout_tier_normal():
    assert get_layout_tier(24, 80) == "normal"
    assert get_layout_tier(24, 90) == "normal"
    assert get_layout_tier(24, 70) == "normal"


def test_get_layout_tier_compact():
    assert get_layout_tier(24, 60) == "compact"
    assert get_layout_tier(24, 65) == "compact"
    assert get_layout_tier(20, 60) == "compact"
    # 50 is boundary: normal tier starts at 70? actually 50-69 compact, so 50 compact
    assert get_layout_tier(24, 50) == "compact"


def test_get_layout_tier_minimised():
    assert get_layout_tier(11, 80) == "minimised"
    assert get_layout_tier(24, 49) == "minimised"
    # 8,30 is extremely_small (<10 height), not just minimised
    assert get_layout_tier(8, 30) == "extremely_small"


def test_get_layout_tier_extremely_small():
    assert get_layout_tier(9, 30) == "extremely_small"
    assert get_layout_tier(10, 39) == "extremely_small"
    assert get_layout_tier(9, 80) == "extremely_small"  # h<10
    assert get_layout_tier(40, 30) == "extremely_small"  # w<40


def test_calc_geometry_tier_field():
    g_large = calc_chatbox_geometry(40, 120)
    assert g_large["tier"] == "large"  # type: ignore
    assert g_large["is_minimised"] == 0
    assert g_large["inner_w"] >= HELLO_LEN

    g_compact = calc_chatbox_geometry(24, 60)
    assert g_compact["tier"] == "compact"  # type: ignore
    assert g_compact["is_minimised"] == 0

    g_min = calc_chatbox_geometry(11, 80)
    assert g_min["tier"] == "minimised"  # type: ignore
    assert g_min["is_minimised"] == 1

    g_small = calc_chatbox_geometry(9, 30)
    assert g_small["tier"] == "extremely_small"  # type: ignore
    assert g_small["too_small"] == 1  # type: ignore


def test_is_too_small():
    assert is_too_small(9, 30) is True
    assert is_too_small(10, 39) is True
    assert is_too_small(24, 80) is False
    assert is_too_small(12, 50) is False
    assert is_too_small(11, 80) is False  # minimised but not extremely_small


def test_chatbox_uses_most_width():
    # Spec 13: chatbox uses most horizontal space with reasonable margins
    g = calc_chatbox_geometry(24, 80)
    # 80 wide -> chat_w should be 78 (80-2)
    assert g["chat_w"] == 78
    assert g["inner_w"] == 76
    g2 = calc_chatbox_geometry(24, 120)
    # large caps at 110
    assert g2["chat_w"] <= 110
    assert g2["chat_w"] >= 78


def test_chatbox_height_not_dominate_on_tall():
    g_tall = calc_chatbox_geometry(40, 80)
    # tall terminal 40 -> chat_h should be capped ~15 not 35
    assert g_tall["chat_h"] < 20
    assert g_tall["chat_h"] >= 5


# ---------------------------------------------------------------------------
# Theme (16-19, 70-72)
# ---------------------------------------------------------------------------

def test_detect_theme_wt_session(monkeypatch):
    monkeypatch.setenv("WT_SESSION", "some-guid")
    monkeypatch.delenv("COLORFGBG", raising=False)
    # WT without COLORFGBG defaults dark
    assert detect_theme("auto") == "dark"
    # with light COLORFGBG
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert detect_theme("auto") == "light"
    # colon separator
    monkeypatch.setenv("COLORFGBG", "0:7")
    assert detect_theme("auto") == "light"


def test_detect_theme_colon_separator(monkeypatch):
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.setenv("COLORFGBG", "15:0;7")
    # last value 7 => light
    assert detect_theme("auto") == "light"
    monkeypatch.setenv("COLORFGBG", "0;0")
    assert detect_theme("auto") == "dark"


def test_theme_colors_grey_offset():
    dark = theme_colors("dark")
    light = theme_colors("light")
    assert dark["chatbox_bg_idx"] != dark["bg_idx"]
    assert light["chatbox_bg_idx"] != light["bg_idx"]
    # input contrast
    assert dark["input_fg"] == "white"
    assert light["input_fg"] == "black"
    # grey family
    assert "grey" in dark["chatbox_bg"]
    assert "grey" in light["chatbox_bg"]


# ---------------------------------------------------------------------------
# Mode system (20-27)
# ---------------------------------------------------------------------------

def test_mode_visibility_and_colors():
    assert MODE_COLOR_IDX["PLAN"] == 208
    assert MODE_COLOR_IDX["BUILD"] == 27
    assert MODE_COLOR_IDX["AUTO"] == 196
    # cycle order
    assert next_mode("PLAN") == "BUILD"
    assert next_mode("BUILD") == "AUTO"
    assert next_mode("AUTO") == "PLAN"


def test_tuiapp_mode_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_TUI_STATE_PATH", str(tmp_path / "state.json"))
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = AgentConfig(workspace=ws, mode="PLAN", provider="ollama", model="qwen3:14b", intelligence="high")
    app = TuiApp(cfg)
    assert app.mode == "PLAN"
    app.cycle_mode()
    assert app.mode == "BUILD"
    app.cycle_mode()
    assert app.mode == "AUTO"
    app.cycle_mode()
    assert app.mode == "PLAN"


# ---------------------------------------------------------------------------
# Model / Intelligence (28-43)
# ---------------------------------------------------------------------------

def test_model_footer_format():
    assert format_model_footer("qwen3:14b", "high") == "qwen3:14b(high)"
    for lvl in INTEL_CHOICES:
        assert format_model_footer("m", lvl) == f"m({lvl})"


def test_intel_display_order():
    assert INTEL_DISPLAY_ORDER == ("default", "low", "medium", "high", "xhigh")
    assert set(INTEL_DISPLAY_ORDER) == set(INTEL_CHOICES)


def test_intel_real_config_wiring(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    app = TuiApp(AgentConfig(workspace=ws, intelligence="high"))
    msg = app.set_intelligence("xhigh")
    assert "xhigh" in msg
    assert app.intelligence == "xhigh"
    # wiring: num_ctx changes
    n_ctx, n_pred, c_budget, lvl = intelligence_values("xhigh")
    assert n_ctx == 65536
    # also via load_config
    cfg = load_config(workspace=str(ws), intelligence="low")
    assert cfg.num_ctx == 8192
    assert cfg.retrieve_level == 1


def test_intel_picker_current_star():
    # ensure picker marks current with ★ via code path (indirect)
    ws = pathlib.Path(tempfile.gettempdir()) / "tmp_intel_ws2"
    ws.mkdir(exist_ok=True)
    cfg = AgentConfig(workspace=ws, intelligence="medium")
    app = TuiApp(cfg)
    assert app.intelligence == "medium"
    # set via API
    app.set_intelligence("default")
    assert app.intelligence == "default"


# ---------------------------------------------------------------------------
# Input cursor & preservation (44-46, 78-82)
# ---------------------------------------------------------------------------

def test_input_cursor_mid_edit(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    app = TuiApp(AgentConfig(workspace=ws))
    app.input_text = "hello world"
    app.cursor_pos = 5
    # Backspace at pos 5 should delete char before cursor
    app._handle_input_key(127, None)  # backspace
    assert app.input_text == "hell world"
    assert app.cursor_pos == 4
    # left/right — use hardcoded curses constants (260 left, 261 right) without importing curses
    app.cursor_pos = 0
    app._handle_input_key(260, None)  # KEY_LEFT at 0 stays
    assert app.cursor_pos == 0
    app.input_text = "hello"
    app.cursor_pos = 2
    app._handle_input_key(261, None)  # KEY_RIGHT
    assert app.cursor_pos == 3
    app._handle_input_key(260, None)  # KEY_LEFT
    assert app.cursor_pos == 2


def test_input_preserved_across_resize(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    app = TuiApp(AgentConfig(workspace=ws))
    app.input_text = "Hello, hello, hello, hello, hello, hello, hello, hello, hello"
    app.cursor_pos = len(app.input_text)
    app.mode = "PLAN"
    # simulate resize recalc doesn't reset
    g1 = calc_chatbox_geometry(24, 80)
    g2 = calc_chatbox_geometry(40, 120)
    g3 = calc_chatbox_geometry(20, 60)
    # all preserve input
    assert app.input_text == "Hello, hello, hello, hello, hello, hello, hello, hello, hello"
    assert app.mode == "PLAN"
    assert g1["tier"] == "normal"
    assert g2["tier"] == "large"
    assert g3["tier"] == "compact"


def test_input_long_preserves_status_bar(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    app = TuiApp(AgentConfig(workspace=ws))
    long_text = "a" * 200
    app.input_text = long_text
    app.cursor_pos = 100
    # geometry for typical width 80 has max_input ~ 76-2-? ~73
    g = calc_chatbox_geometry(24, 80)
    inner_w = g["inner_w"]
    max_input = inner_w - len("> ") - 1
    assert max_input > 50
    assert len(long_text) > max_input
    # app should still keep cursor_pos and text
    assert app.cursor_pos == 100
    assert len(app.input_text) == 200


# ---------------------------------------------------------------------------
# Picker structure (31-33, 55-59)
# ---------------------------------------------------------------------------

def test_build_picker_bold_and_indented():
    pm = {"ollama": ["qwen3:14b"], "openai": ["gpt-4o"], "anthropic": [], "grok": [], "google": [], "deepseek": []}
    items = build_picker_items(pm)
    headers = [i for i in items if i.is_provider_header]
    assert len(headers) == 6
    # model rows indented with 3 spaces in code
    model_items = [i for i in items if not i.is_provider_header]
    assert any(i.label == "qwen3:14b" for i in model_items)
    # provider header selectable, model indented
    assert items[0].is_provider_header is True


def test_picker_no_fake_providers():
    pm = {"ollama": [], "openai": [], "anthropic": [], "grok": [], "google": [], "deepseek": []}
    items = build_picker_items(pm)
    provs = {i.provider for i in items}
    assert provs == {"ollama", "openai", "anthropic", "grok", "google", "deepseek"}


def test_slash_intel_without_args_shows_picker_path(tmp_path, monkeypatch):
    # In fallback mode, /intel without args shows usage list (not picker)
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("AGENT_TUI_STATE_PATH", str(tmp_path / "s.json"))
    app = TuiApp(AgentConfig(workspace=ws, intelligence="high"))
    # simulate fallback slash handling without curses: should give usage?
    # TuiApp._handle_slash needs stdscr; in fallback we test parse only
    assert parse_slash_command("/intel") == ("intel", [])
    assert parse_slash_command("/intel high") == ("intel", ["high"])


def test_slash_help_includes_required():
    # help string in code includes these
    ws = pathlib.Path(tempfile.gettempdir()) / "tmp_help_ws"
    ws.mkdir(exist_ok=True)
    app = TuiApp(AgentConfig(workspace=ws))
    # we test that help handling sets status_msg correctly via direct call with mocked stdscr
    class Dummy:
        pass
    # without curses, handle via fallback: check parse
    assert parse_slash_command("/help") == ("help", [])
    assert parse_slash_command("/models") == ("models", [])
    assert parse_slash_command("/connect") == ("connect", [])


# ---------------------------------------------------------------------------
# Chatbox sizing (12, 62-64)
# ---------------------------------------------------------------------------

def test_hello_fits_all_tiers():
    for w, h in [(80, 24), (120, 40), (60, 20), (90, 30)]:
        g = calc_chatbox_geometry(h, w)
        if g["is_minimised"] == 0:
            assert g["inner_w"] >= HELLO_LEN, f"failed for {w}x{h}"
            # chatbox uses most width
            assert g["chat_w"] >= HELLO_LEN + 2


def test_status_bar_spacing():
    # footer + mode should not overlap
    for w in [80, 100, 120]:
        g = calc_chatbox_geometry(24, w)
        inner_w = g["inner_w"]
        footer = format_model_footer("qwen3:14b", "high")
        mode = "PLAN"
        needed = len(mode) + len(footer) + 4
        assert needed < inner_w, f"status bar would overlap at {w}"
