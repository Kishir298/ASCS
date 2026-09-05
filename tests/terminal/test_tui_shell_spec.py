"""Shell-spec backfill: strict /models scope, /connect validation, layout helpers.

Covers the gaps found in the TUI shell re-check pass:
  - build_scoped_picker_items (active provider only)
  - validate_connect_inputs (strict pre-flight)
  - COMFORTABLE_TIER / is_comfortable_layout alias
  - chatbox_bottom_layout (mode bottom-left, model(intel) right, no overlap)
  - slash_menu_text + bare "/" menu via _handle_slash
  - /models empty-state (no models -> /connect hint, no picker)
  - theme override wins over env detection
  - 63-col chatbox floor kept at compact widths
"""

import pytest

from agent.config import AgentConfig
from agent.tui import (
    COMFORTABLE_TIER,
    HELLO_LEN,
    MIN_CHATBOX_W,
    SLASH_COMMANDS,
    TuiApp,
    build_scoped_picker_items,
    calc_chatbox_geometry,
    chatbox_bottom_layout,
    detect_theme,
    format_model_footer,
    get_layout_tier,
    is_comfortable_layout,
    slash_menu_text,
    validate_connect_inputs,
)


# -- scoped picker ------------------------------------------------------------


def test_scoped_picker_only_active_provider():
    pm = {"ollama": ["a", "b"], "openai": ["c"], "anthropic": ["d"]}
    items = build_scoped_picker_items(pm, "ollama")
    assert items[0].is_provider_header and items[0].provider == "ollama"
    assert [i.label for i in items if not i.is_provider_header] == ["a", "b"]
    assert all(i.provider == "ollama" for i in items)


def test_scoped_picker_empty_models_keeps_header():
    items = build_scoped_picker_items({"ollama": []}, "ollama")
    assert len(items) == 1 and items[0].is_provider_header


def test_scoped_picker_unknown_provider_falls_back_to_ollama():
    items = build_scoped_picker_items({"ollama": ["m"]}, "  ")
    assert items[0].provider == "ollama"


# -- connect validation --------------------------------------------------------


def test_connect_rejects_empty_url():
    assert validate_connect_inputs("ollama", "", "") is not None
    assert validate_connect_inputs("openai", "   ", "k") is not None


def test_connect_rejects_bad_scheme():
    err = validate_connect_inputs("ollama", "localhost:11434", "")
    assert err is not None and "http" in err


def test_connect_ollama_localhost_no_key_ok():
    assert validate_connect_inputs("ollama", "http://localhost:11434", "") is None


def test_connect_cloud_default_url_requires_key():
    err = validate_connect_inputs("openai", "https://api.openai.com", "")
    assert err is not None and "API key" in err


def test_connect_cloud_local_endpoint_no_key_ok():
    # LM Studio / local OpenAI-compat endpoint masquerading as a cloud provider
    assert validate_connect_inputs("openai", "http://localhost:1234/v1", "") is None


def test_connect_cloud_with_key_ok():
    assert validate_connect_inputs("anthropic", "https://api.anthropic.com", "sk-x") is None


# -- tier naming + geometry floor -----------------------------------------------


def test_comfortable_tier_is_normal():
    assert COMFORTABLE_TIER == "normal"
    assert is_comfortable_layout("normal") is True
    assert is_comfortable_layout(get_layout_tier(24, 80)) is True
    assert is_comfortable_layout("compact") is False


def test_compact_widths_keep_63_col_floor():
    for w in (50, 55, 60, 63, 65, 69):
        g = calc_chatbox_geometry(24, w)
        assert g["tier"] == "compact"
        assert g["is_minimised"] == 0
        assert g["chat_w"] >= MIN_CHATBOX_W  # 63: never shrunk below minimum
        assert g["inner_w"] >= HELLO_LEN


# -- bottom-line layout ----------------------------------------------------------


@pytest.mark.parametrize("mode", ["PLAN", "BUILD", "AUTO"])
def test_bottom_layout_mode_left_footer_right_no_overlap(mode):
    chat_w, inner_w = 78, 76
    mode_str, footer, fx = chatbox_bottom_layout(mode, "qwen3-coder:30b", "high", chat_w, inner_w)
    assert mode_str == f" {mode} "
    assert footer == "qwen3-coder:30b(high)"
    assert fx == chat_w - len(footer) - 3
    mode_end = 2 + len(mode_str)
    assert mode_end < fx  # never overlap


def test_bottom_layout_footer_format_matches_helper():
    for lvl in ("low", "medium", "high", "xhigh", "default"):
        _, footer, _ = chatbox_bottom_layout("AUTO", "m", lvl, 78, 76)
        assert footer == format_model_footer("m", lvl) == f"m({lvl})"


def test_bottom_layout_compact_truncates_long_footer():
    _, footer, fx = chatbox_bottom_layout("PLAN", "x" * 60, "xhigh", 65, 63, tier="compact")
    assert len(footer) + 4 <= 63 or footer.endswith("…")
    assert fx >= 2 + len(" PLAN ")


# -- slash menu ------------------------------------------------------------------


def test_slash_menu_lists_all_three_commands():
    text = slash_menu_text()
    for name in ("/models", "/connect", "/intel"):
        assert name in text
    assert len(SLASH_COMMANDS) == 3


def test_bare_slash_shows_menu(tmp_path):
    app = TuiApp(AgentConfig(workspace=tmp_path))

    class Dummy:
        def getmaxyx(self):
            return (24, 80)

    app._handle_slash(Dummy(), "/")
    assert any("/models" in m.get("content", "") for m in app.messages)
    assert any("/connect" in m.get("content", "") for m in app.messages)
    assert any("/intel" in m.get("content", "") for m in app.messages)


def test_models_empty_state_points_to_connect(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent.providers.list_all_providers_with_models",
        lambda timeout=2, use_cache=True: {"ollama": []},
    )
    app = TuiApp(AgentConfig(workspace=tmp_path, provider="ollama", model="qwen3-coder:30b"))

    class Dummy:
        def getmaxyx(self):
            return (24, 80)

    app._handle_slash(Dummy(), "/models")
    assert "/connect" in app.status_msg
    assert app.model == "qwen3-coder:30b"  # unchanged, picker never opened


# -- theme override ---------------------------------------------------------------


def test_explicit_theme_overrides_env_detection(monkeypatch):
    monkeypatch.setenv("WT_SESSION", "some-guid")
    monkeypatch.setenv("COLORFGBG", "15;0")  # dark hint
    assert detect_theme("light") == "light"
    assert detect_theme("dark") == "dark"
