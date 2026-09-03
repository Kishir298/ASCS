"""Integration tests for real interactive TUI — no live Ollama required.

Covers:
  - UI state: mode defaults, TAB cycles, intel/model/theme
  - Commands: /help /models /connect /intel /status /clear /history /experiences /tasks /check /quit
  - Terminal: resize tiers, small terminal guards, clean exit
  - Integration: TaskRunner wiring, cancellation, no fake HELLO/queue
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from agent.config import AgentConfig, DEFAULT_MODEL, intelligence_values, load_config
from agent.tui import (
    HELLO_TEXT,
    HELLO_LEN,
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


# -- UI state ---------------------------------------------------------------

def test_tui_defaults_are_real_model(tmp_path):
    cfg = AgentConfig(workspace=tmp_path, model="qwen3-coder:30b", intelligence="default")
    app = TuiApp(cfg)
    assert app.model == "qwen3-coder:30b"
    assert app.mode in MODE_ORDER
    assert app.intelligence == "default"
    assert DEFAULT_MODEL == "qwen3-coder:30b"


def test_tab_cycles_plan_build_auto(tmp_path):
    cfg = AgentConfig(workspace=tmp_path, mode="PLAN")
    app = TuiApp(cfg)
    assert app.mode == "PLAN"
    app.cycle_mode()
    assert app.mode == "BUILD"
    app.cycle_mode()
    assert app.mode == "AUTO"
    app.cycle_mode()
    assert app.mode == "PLAN"


def test_next_mode_helper():
    assert next_mode("PLAN") == "BUILD"
    assert next_mode("BUILD") == "AUTO"
    assert next_mode("AUTO") == "PLAN"
    assert next_mode("unknown") == "PLAN"


def test_intelligence_levels_all_work(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    for lvl in ("low", "medium", "high", "xhigh", "default"):
        validated = validate_intel(lvl)
        assert validated == lvl
        n_ctx, n_pred, budget, rlvl = intelligence_values(lvl)
        assert n_ctx > 0 and n_pred > 0
    # xhigh is max-chunking 131k
    assert intelligence_values("xhigh")[0] == 131072
    assert intelligence_values("default")[0] == 65536


def test_model_footer_format():
    assert format_model_footer("qwen3-coder:30b", "high") == "qwen3-coder:30b(high)"
    assert format_model_footer("qwen2.5-coder:14b", "low") == "qwen2.5-coder:14b(low)"


def test_theme_auto_resolves(tmp_path):
    assert detect_theme("light") == "light"
    assert detect_theme("dark") == "dark"
    assert detect_theme("auto") in ("light", "dark")
    dark = theme_colors("dark")
    light = theme_colors("light")
    assert dark["chatbox_bg"] != dark["bg"]
    assert light["chatbox_bg"] != light["bg"]
    assert dark["input_fg"] == "white"
    assert light["input_fg"] == "black"


# -- No fake preview --------------------------------------------------------

def test_no_hello_in_empty_render(tmp_path):
    """TUI must not render HELLO_TEXT as fake demo — HELLO_LEN constant stays for tests but not displayed."""
    cfg = AgentConfig(workspace=tmp_path)
    app = TuiApp(cfg)
    # New app has welcome system message, not HELLO_TEXT
    assert len(app.messages) == 0  # before draw
    # Simulate what _draw would do for empty messages: welcome
    app._add_system("Welcome")
    assert app.messages[0]["content"] != HELLO_TEXT
    assert HELLO_TEXT == "Hello, hello, hello, hello, hello, hello, hello, hello, hello"
    assert HELLO_LEN == 61


def test_no_queued_fake_status(tmp_path):
    cfg = AgentConfig(workspace=tmp_path)
    app = TuiApp(cfg)
    # Status should be help line, not queued fake
    assert "queued:" not in app.status_msg
    assert "/help" in app.status_msg


# -- Input control ----------------------------------------------------------

def test_input_editable_home_end_arrows(tmp_path):
    cfg = AgentConfig(workspace=tmp_path)
    app = TuiApp(cfg)
    app.input_text = "hello world"
    app.cursor_pos = 5
    # Home
    app._handle_input_key(262, None)  # KEY_HOME
    assert app.cursor_pos == 0
    # End
    app._handle_input_key(360, None)  # KEY_END
    assert app.cursor_pos == len("hello world")
    # Left
    app._handle_input_key(260, None)
    assert app.cursor_pos == len("hello world") - 1
    # Right
    app._handle_input_key(261, None)
    assert app.cursor_pos == len("hello world")
    # Ctrl+A / Ctrl+E
    app.cursor_pos = 3
    app._handle_input_key(1, None)
    assert app.cursor_pos == 0
    app._handle_input_key(5, None)
    assert app.cursor_pos == len(app.input_text)


def test_input_typing_and_backspace(tmp_path):
    cfg = AgentConfig(workspace=tmp_path)
    app = TuiApp(cfg)
    for ch in "hi":
        app._handle_input_key(ord(ch), None)
    assert app.input_text == "hi"
    assert app.cursor_pos == 2
    app._handle_input_key(127, None)  # backspace
    assert app.input_text == "h"
    app._handle_input_key(330, None)  # delete at end does nothing
    assert app.input_text == "h"


# -- Responsive tiers -------------------------------------------------------

def test_resize_tiers():
    assert get_layout_tier(40, 120) == "large"
    assert get_layout_tier(24, 80) == "normal"
    assert get_layout_tier(20, 60) == "compact"
    assert get_layout_tier(11, 40) == "minimised"
    assert get_layout_tier(9, 30) == "extremely_small"
    assert is_too_small(9, 30) is True
    assert is_minimised(24, 80) is False
    assert calc_chatbox_geometry(24, 80)["is_minimised"] == 0
    assert calc_chatbox_geometry(9, 30)["too_small"] == 1


# -- Slash commands (real, not fake) ---------------------------------------

def test_help_lists_real_commands(tmp_path):
    cfg = AgentConfig(workspace=tmp_path)
    app = TuiApp(cfg)
    app._handle_slash(None, "/help")
    assert any("/models" in m["content"] for m in app.messages)
    assert any("/intel" in m["content"] for m in app.messages)


def test_intel_changes_real_state(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_TUI_STATE_PATH", str(tmp_path / "state.json"))
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = AgentConfig(workspace=ws, intelligence="default")
    app = TuiApp(cfg)
    app._handle_slash(None, "/intel high")
    assert app.intelligence == "high"
    assert "high" in app.messages[-1]["content"]
    # Invalid
    before = app.intelligence
    app._handle_slash(None, "/intel nonsense")
    assert app.intelligence == before  # unchanged


def test_status_shows_real_state(tmp_path):
    cfg = AgentConfig(workspace=tmp_path)
    app = TuiApp(cfg)
    app._handle_slash(None, "/status")
    txt = app.messages[-1]["content"]
    assert "Provider:" in txt
    assert "Model:" in txt
    assert "qwen3-coder:30b" in txt or "qwen2.5-coder" in txt or "Model:" in txt


def test_clear_preserves_config(tmp_path):
    cfg = AgentConfig(workspace=tmp_path, intelligence="high")
    app = TuiApp(cfg)
    app._add_message("user", "hello")
    app._add_message("assistant", "hi")
    assert len(app.messages) >= 2
    app._handle_slash(None, "/clear")
    # After clear we add system "Cleared" so 1 message remains, but user history cleared
    assert len(app.messages) == 1
    assert app.intelligence == "high"


def test_history_and_experiences(tmp_path):
    cfg = AgentConfig(workspace=tmp_path)
    app = TuiApp(cfg)
    app._add_message("user", "first")
    app._add_message("assistant", "second")
    app._handle_slash(None, "/history")
    assert "first" in app.messages[-1]["content"]
    # experiences should not crash even if no store
    app._handle_slash(None, "/experiences")
    assert len(app.messages) >= 2


def test_tasks_and_check(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = AgentConfig(workspace=ws)
    app = TuiApp(cfg)
    app._handle_slash(None, "/tasks")
    assert len(app.messages) >= 1
    app._handle_slash(None, "/check")
    # doctor produces PASS/FAIL lines
    assert len(app.messages) >= 2


def test_quit_sets_flag(tmp_path):
    cfg = AgentConfig(workspace=tmp_path)
    app = TuiApp(cfg)
    assert not app.should_quit
    app._handle_slash(None, "/quit")
    assert app.should_quit
    app2 = TuiApp(cfg)
    app2._handle_slash(None, "/q")
    assert app2.should_quit


def test_parse_slash():
    assert parse_slash_command("/intel high") == ("intel", ["high"])
    assert parse_slash_command("/models") == ("models", [])
    assert parse_slash_command("hello") == ("", [])


# -- Model picker (mocked, not hardcoded) ---------------------------------

def test_build_picker_uses_real_provider_order():
    pm = {"ollama": ["qwen3-coder:30b"], "openai": ["gpt-4o"], "anthropic": [], "grok": [], "google": [], "deepseek": []}
    items = build_picker_items(pm)
    headers = [i for i in items if i.is_provider_header]
    assert len(headers) == 6
    assert items[0].provider == "ollama"
    assert PINK_BG_IDX in (200, 201, 205, 211, 213, 219)


def test_models_picker_mocked(tmp_path, monkeypatch):
    # Mock provider fetch to avoid network
    import agent.tui as tui_mod
    called = {}

    def fake_list_all(timeout=2, use_cache=True):
        called["hit"] = True
        return {"ollama": ["qwen3-coder:30b", "qwen2.5-coder:14b"], "openai": []}

    monkeypatch.setattr("agent.providers.list_all_providers_with_models", fake_list_all)
    cfg = AgentConfig(workspace=tmp_path, provider="ollama", model="qwen3-coder:30b")
    app = TuiApp(cfg)
    # Mock picker to auto-select second model
    monkeypatch.setattr(app, "_run_picker", lambda stdscr, pm: ("ollama", "qwen2.5-coder:14b"))
    # Need a dummy stdscr for call
    class Dummy:
        def getmaxyx(self):
            return (24, 80)
    app._handle_slash(Dummy(), "/models")
    assert app.model == "qwen2.5-coder:14b"
    assert app.provider == "ollama"


# -- Cancellation + streaming wiring ---------------------------------------

def test_cancel_when_not_busy(tmp_path):
    cfg = AgentConfig(workspace=tmp_path)
    app = TuiApp(cfg)
    app._cancel_running()
    assert "Nothing to cancel" in app.status_msg or "cancel" in app.status_msg.lower()


def test_cancel_when_busy_mocked(tmp_path, monkeypatch):
    cfg = AgentConfig(workspace=tmp_path)
    app = TuiApp(cfg)
    class FakeRunner:
        busy = True
        def __init__(self):
            self.cancel_called = False
        def cancel(self):
            self.cancel_called = True
    fake = FakeRunner()
    app._runner = fake
    app._cancel_running()
    assert fake.cancel_called


def test_start_task_uses_real_runner(tmp_path, monkeypatch):
    # Fake hub/runner without unittest.mock (AppControl blocks asyncio)
    class FakeHub:
        def history(self):
            return []
        def clear(self):
            pass
    class FakeRunner:
        busy = False
        result = None
        def __init__(self, *a, **kw):
            self.start_called = False
        def start(self, task, mode=None):
            self.start_called = True
            return True
    fake_hub = FakeHub()
    fake_runner = FakeRunner()
    # Patch web module classes via monkeypatch
    import agent.web as web_mod
    monkeypatch.setattr(web_mod, "EventHub", lambda: fake_hub)
    monkeypatch.setattr(web_mod, "TaskRunner", lambda *a, **kw: fake_runner)
    cfg = AgentConfig(workspace=tmp_path)
    app = TuiApp(cfg)
    app._hub = fake_hub
    ok = app._start_task("hello")
    assert ok is True
    assert app.messages[0]["role"] == "user"
    assert "hello" in app.messages[0]["content"]


def test_poll_runner_adds_assistant_on_completion(tmp_path):
    cfg = AgentConfig(workspace=tmp_path)
    app = TuiApp(cfg)
    class FakeEvent:
        type = "agent_completed"
        message = "Done from model"
    class FakeHub:
        def history(self):
            return [FakeEvent()]
    class FakeRunner:
        busy = False
        class Result:
            status = "completed"
            summary = "Done from model"
            error = ""
        result = Result()
    app._hub = FakeHub()
    app._last_event_count = 0
    app._runner = FakeRunner()
    app._poll_runner()
    assert any("Done from model" in m["content"] for m in app.messages)


# -- Clean exit -------------------------------------------------------------

def test_run_fallback_returns_error(tmp_path):
    cfg = AgentConfig(workspace=tmp_path)
    app = TuiApp(cfg)
    rc = app.run_fallback()
    assert rc == 1


def test_live_config_reflects_ui_state(tmp_path):
    cfg = AgentConfig(workspace=tmp_path, mode="PLAN", model="qwen2.5-coder:14b", intelligence="low")
    app = TuiApp(cfg)
    app.mode = "BUILD"
    app.model = "qwen3-coder:30b"
    app.intelligence = "xhigh"
    live = app._live_config()
    assert live.mode == "BUILD"
    assert live.model == "qwen3-coder:30b"
    assert live.num_ctx == 131072
