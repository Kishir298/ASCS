"""Combined CLI interface tests — union of CLI-related suites.

Sources (kept as duplicates for rollback / safety):
  - tests/test_cli.py (4 tests) — lightweight CLI parser / main entry
  - tests/test_tui_spec.py (14 tests) — TUI spec verification
  - tests/test_shell_redesign.py (28 tests) — shell redesign tiers/cursor/theme/pickers

Total: 46 tests, no name collisions (verified via ast).
This file deduplicates top-level imports and preserves each test body verbatim
with section headers for traceability. Future changes to sources must be
re-synced manually or by re-running the combine plan.

Inline function-local imports (e.g. `from agent.tui import MODE_COLOR_IDX`
inside test bodies) are preserved verbatim to avoid behavioural drift.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from agent.config import (
    INTELLIGENCE_LEVELS,
    INTELLIGENCE_MAP,
    PROVIDER_NAMES,
    AgentConfig,
    intelligence_values,
    load_config,
    load_tui_state,
    save_tui_state,
)
from agent.main import build_parser, main
from agent.providers import (
    PROVIDER_NAMES as P_NAMES,
    get_ollama_compat_models,
    is_ollama_available,
    list_all_providers_with_models,
    list_models_for_provider,
)
from agent.tui import (
    HELLO_LEN,
    HELLO_TEXT,
    INTEL_CHOICES,
    INTEL_DISPLAY_ORDER,
    MODE_COLOR_IDX,
    MODE_ORDER,
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

# Alias used by test_tui_spec.py
tui_build_picker_items = build_picker_items

# =============================================================================
# Section 1 — tests/test_cli.py (4 tests)
# Lightweight CLI tests that avoid needing a live Ollama server.
# =============================================================================


def test_parser_has_expected_flags():
    p = build_parser()
    argv = [
        "--workspace",
        ".",
        "--model",
        "m",
        "--base-url",
        "u",
        "--max-iterations",
        "5",
        "--verbose",
        "--safe",
        "do the task",
    ]
    ns = p.parse_args(argv)
    assert ns.workspace == "."
    assert ns.model == "m"
    assert ns.base_url == "u"
    assert ns.max_iterations == 5
    assert ns.verbose is True
    assert ns.safe is True
    assert ns.task == "do the task"


def test_safe_and_auto_conflict_exits_2(capsys):
    rc = main(["--safe", "--auto", "--workspace", ".", "task"])
    assert rc == 2
    out = capsys.readouterr().err
    assert "Cannot use --safe and --auto" in out


def test_version_flag(capsys):
    with __import__("pytest").raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "risa" in out


def test_unknown_argument_is_error():
    import pytest

    with pytest.raises(SystemExit):
        main(["--definitely-not-a-flag"])


# =============================================================================
# Section 2 — tests/test_tui_spec.py (14 tests)
# Spec verification for the TUI CLI interface.
# =============================================================================


def test_hello_text_exact():
    assert HELLO_TEXT == "Hello, hello, hello, hello, hello, hello, hello, hello, hello"
    assert HELLO_LEN == 61
    assert len(HELLO_TEXT) == 61


def test_chatbox_fits_hello_exactly():
    # 24x80 is typical; inner must be >=61
    g = calc_chatbox_geometry(24, 80)
    assert g["inner_w"] >= HELLO_LEN
    # exact minimal terminal that fits hello: width 63 => inner 61
    g2 = calc_chatbox_geometry(24, 63)
    assert g2["inner_w"] == HELLO_LEN
    assert g2["chat_w"] == HELLO_LEN + 2
    # smaller => minimised per spec (w<50 => inner 0), hello truncated via fallback
    g3 = calc_chatbox_geometry(24, 40)
    assert g3["is_minimised"] == 1
    assert g3["inner_w"] == 0


def test_chatbox_minimised_on_small_terminal():
    assert is_minimised(8, 30) is True
    assert is_minimised(9, 40) is True
    assert is_minimised(24, 80) is False
    assert is_minimised(12, 50) is False
    # exactly at threshold
    assert is_minimised(11, 80) is True  # h<12 => minimised
    assert is_minimised(24, 49) is True  # w<50 => minimised


def test_tab_cycles_plan_build_auto():
    assert list(MODE_ORDER) == ["PLAN", "BUILD", "AUTO"]
    assert next_mode("PLAN") == "BUILD"
    assert next_mode("BUILD") == "AUTO"
    assert next_mode("AUTO") == "PLAN"
    # unknown => PLAN
    assert next_mode("BANANA") == "PLAN"
    assert next_mode("") == "PLAN"


def test_mode_colors_spec():
    # colors are orange/blue/red as per spec, pink highlight
    from agent.tui import MODE_COLOR_IDX

    assert MODE_COLOR_IDX["PLAN"] == 208  # orange
    assert MODE_COLOR_IDX["BUILD"] == 27  # blue
    assert MODE_COLOR_IDX["AUTO"] == 196  # red
    assert PINK_BG_IDX in (200, 201, 205, 211, 213, 219)  # any pink


def test_theme_background_and_input_contrast():
    # dark: black bg, white input
    c_dark = theme_colors("dark")
    assert c_dark["bg"] == "black"
    assert c_dark["input_fg"] == "white"
    # light: white bg, black input
    c_light = theme_colors("light")
    assert c_light["bg"] == "white"
    assert c_light["input_fg"] == "black"
    # chatbox slightly offset from bg
    assert c_dark["chatbox_bg"] != c_dark["bg"]
    assert c_light["chatbox_bg"] != c_light["bg"]
    # auto resolves to dark by default when no env
    assert detect_theme("auto") in ("light", "dark")
    assert detect_theme("dark") == "dark"
    assert detect_theme("light") == "light"


def test_footer_format_model_intelligence():
    assert format_model_footer("qwen3-coder:30b", "high") == "qwen3-coder:30b(high)"
    assert format_model_footer("gpt-4o", "xhigh") == "gpt-4o(xhigh)"
    # same colour as input (verified via theme_colors)
    tc = theme_colors("dark")
    assert tc["input_fg"] == "white"
    # footer uses same fg; we don't test curses pair here, just string


def test_intelligence_levels_both_window_and_retrieval():
    # Spec: intelligence changes tokens per request with 5 modes
    assert set(INTELLIGENCE_LEVELS) == {"low", "medium", "high", "xhigh", "default"}
    # each level maps to (num_ctx, num_predict, budget, level)
    for lvl in INTELLIGENCE_LEVELS:
        n_ctx, n_pred, budget, lvl_num = intelligence_values(lvl)
        assert n_ctx > 0 and n_pred > 0 and budget > 0
        assert 1 <= lvl_num <= 4
    # verify both aspects: low vs xhigh
    low = intelligence_values("low")
    xhigh = intelligence_values("xhigh")
    assert low[0] < xhigh[0]  # num_ctx
    assert low[3] < xhigh[3]  # retrieve level
    # default alias should equal high's window (65k bump for 30b)
    assert intelligence_values("default") == intelligence_values("high") or intelligence_values("default")[0] == 65536
    # config integration: intelligence sets both
    import tempfile, pathlib

    tmp = pathlib.Path(tempfile.gettempdir()) / "tmp_tui_intel_ws"
    tmp.mkdir(exist_ok=True)
    c_low = load_config(workspace=str(tmp), intelligence="low")
    c_xhigh = load_config(workspace=str(tmp), intelligence="xhigh")
    assert c_low.num_ctx == 8192 and c_low.num_predict == 2048 and c_low.retrieve_level == 1
    assert c_xhigh.num_ctx == 131072 and c_xhigh.num_predict == 32768 and c_xhigh.retrieve_level == 4
    assert c_low.context_budget_chars == 30000
    assert c_xhigh.context_budget_chars == 140000


def test_provider_list_includes_all_majors_and_ollama_always():
    assert set(P_NAMES) == {"ollama", "openai", "anthropic", "grok", "google", "deepseek"}
    assert set(PROVIDER_NAMES) == set(P_NAMES)
    # ollama provider always present
    assert "ollama" in PROVIDER_NAMES


def test_models_per_provider_and_empty_when_no_key():
    # Without keys, cloud providers return empty list (per spec: empty, not error)
    for p in ("openai", "anthropic", "grok", "google", "deepseek"):
        models = list_models_for_provider(p, timeout=1, use_cache=False)
        assert models == [], f"{p} should be [] without API key"
    # Ollama offline returns [] but not crash, still Ollama-compatible
    assert isinstance(list_models_for_provider("ollama", timeout=1, use_cache=False), list)


def test_provider_picker_bold_and_pink_highlight_structure():
    # Provider header should be bold; we verify build_picker_items creates correct structure
    provider_models = {
        "ollama": ["qwen3-coder:30b", "llama3:8b"],
        "openai": ["gpt-4o"],
        "anthropic": [],
        "grok": [],
        "google": [],
        "deepseek": [],
    }
    items = tui_build_picker_items(provider_models)
    # headers are bold
    headers = [i for i in items if i.is_provider_header]
    assert len(headers) == 6
    assert [h.provider for h in headers] == list(PROVIDER_NAMES)
    # openai header + 1 model, anthropic header only (empty)
    openai_items = [i for i in items if i.provider == "openai"]
    assert len(openai_items) == 2  # header + 1 model
    anthropic_items = [i for i in items if i.provider == "anthropic"]
    assert len(anthropic_items) == 1  # only header, no models -> empty per spec
    # model rows are indented and not bold, but selected row gets pink highlight (verified in tui code via PINK_BG_IDX)
    assert any(i.kind == "model" for i in items)


def test_persistence_across_restarts(tmp_path, monkeypatch):
    state_path = tmp_path / "tui_state.json"
    monkeypatch.setenv("AGENT_TUI_STATE_PATH", str(state_path))
    # save state as TUI would
    save_tui_state({"provider": "openai", "model": "gpt-4o", "intelligence": "high", "theme": "dark"})
    data = load_tui_state()
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4o"
    # load_config should pick up persisted values when no env/override
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = load_config(workspace=str(ws))
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o"
    assert cfg.intelligence == "high"
    # explicit override wins over persisted
    cfg2 = load_config(workspace=str(ws), provider="ollama", intelligence="low")
    assert cfg2.provider == "ollama"
    assert cfg2.intelligence == "low"


def test_slash_commands_parse():
    assert parse_slash_command("/models") == ("models", [])
    assert parse_slash_command("/connect") == ("connect", [])
    assert parse_slash_command("/intel high") == ("intel", ["high"])
    assert parse_slash_command("/intel xhigh") == ("intel", ["xhigh"])
    assert parse_slash_command("/help") == ("help", [])
    assert parse_slash_command("not a slash") == ("", [])
    # validate intel
    assert validate_intel("low") == "low"
    assert validate_intel("medium") == "medium"
    assert validate_intel("high") == "high"
    assert validate_intel("xhigh") == "xhigh"
    assert validate_intel("default") == "default"
    with pytest.raises(ValueError):
        validate_intel("ultra")


def test_ollama_compatible_guarantee():
    # Even if cloud providers fail, Ollama is fallback and app never crashes
    from agent.providers import get_ollama_compat_models, is_ollama_available

    # get_ollama_compat_models for unknown provider should fallback to ollama list (empty offline but not error)
    models, is_fallback = get_ollama_compat_models("openai", timeout=1)
    assert isinstance(models, list)
    # is_ollama_available should be bool, not raise
    assert isinstance(is_ollama_available(timeout=1), bool)


# =============================================================================
# Section 3 — tests/test_shell_redesign.py (28 tests)
# Shell redesign verification — tiers, cursor, theme, pickers, resize.
# =============================================================================


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
    cfg = AgentConfig(workspace=ws, mode="PLAN", provider="ollama", model="qwen3-coder:30b", intelligence="high")
    app = TuiApp(cfg)
    assert app.mode == "PLAN"
    app.cycle_mode()
    assert app.mode == "BUILD"
    app.cycle_mode()
    assert app.mode == "AUTO"
    app.cycle_mode()
    assert app.mode == "PLAN"


def test_model_footer_format():
    assert format_model_footer("qwen3-coder:30b", "high") == "qwen3-coder:30b(high)"
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
    # wiring: num_ctx changes (xhigh bumped for 30b)
    n_ctx, n_pred, c_budget, lvl = intelligence_values("xhigh")
    assert n_ctx == 131072
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


def test_build_picker_bold_and_indented():
    pm = {"ollama": ["qwen3-coder:30b"], "openai": ["gpt-4o"], "anthropic": [], "grok": [], "google": [], "deepseek": []}
    items = build_picker_items(pm)
    headers = [i for i in items if i.is_provider_header]
    assert len(headers) == 6
    # model rows indented with 3 spaces in code
    model_items = [i for i in items if not i.is_provider_header]
    assert any(i.label == "qwen3-coder:30b" for i in model_items)
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
        footer = format_model_footer("qwen3-coder:30b", "high")
        mode = "PLAN"
        needed = len(mode) + len(footer) + 4
        assert needed < inner_w, f"status bar would overlap at {w}"
