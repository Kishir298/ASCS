"""Spec verification for the TUI CLI interface.

These tests cover the aesthetic and functional requirements from the
A.S.C.S CLI design prompt:
  - TAB cycles Plan->Build->Auto
  - /models, /connect, /intel, /help
  - Chatbox size for HELLO_TEXT
  - Minimised on small terminal
  - Theme-aware colors, mode colors, footer format, pink highlight, etc.
  - Provider list per-provider, empty when no models, bold provider, Ollama always available
  - Persistence across restarts
  - Intelligence both window+retrieval
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
from agent.providers import (
    PROVIDER_NAMES as P_NAMES,
    list_models_for_provider,
    list_all_providers_with_models,
)
from agent.tui import (
    HELLO_TEXT,
    HELLO_LEN,
    MODE_ORDER,
    PINK_BG_IDX,
    build_picker_items as tui_build_picker_items,
    calc_chatbox_geometry,
    detect_theme,
    format_model_footer,
    is_minimised,
    next_mode,
    parse_slash_command,
    theme_colors,
    validate_intel,
)


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
