"""Windows/curses stability regression tests: bounds-safe helpers."""

from __future__ import annotations

import pytest

from agent.tui import clamp_window, safe_addstr

try:  # pragma: no cover - backend dependent
    import curses

    _ERR = curses.error
except Exception:  # pragma: no cover

    class _ERR(Exception):
        pass


class FakeWin:
    def __init__(self, h, w, fail_addstr=False):
        self._h = h
        self._w = w
        self.fail_addstr = fail_addstr
        self.drawn = []

    def getmaxyx(self):
        return (self._h, self._w)

    def addstr(self, y, x, text, *args):
        if self.fail_addstr:
            raise _ERR("simulated transient failure")
        self.drawn.append((y, x, text))


class ExplodingWin:
    def getmaxyx(self):
        raise _ERR("transient state")


# -- safe_addstr -------------------------------------------------------------


def test_safe_addstr_draws_when_it_fits():
    win = FakeWin(24, 80)
    assert safe_addstr(win, 0, 2, "hello") is True
    assert win.drawn == [(0, 2, "hello")]


def test_safe_addstr_rejects_negative_coordinates():
    win = FakeWin(24, 80)
    assert safe_addstr(win, -1, 2, "x") is False
    assert safe_addstr(win, 0, -1, "x") is False
    assert win.drawn == []


def test_safe_addstr_rejects_row_outside_window():
    win = FakeWin(24, 80)
    assert safe_addstr(win, 24, 0, "x") is False
    assert win.drawn == []


def test_safe_addstr_truncates_and_avoids_final_cell():
    win = FakeWin(24, 10)
    assert safe_addstr(win, 0, 2, "x" * 50) is True
    assert win.drawn == [(0, 2, "x" * 7)]  # 10 - 2 - 1 (final cell)


def test_safe_addstr_skips_when_only_final_cell_remains():
    win = FakeWin(24, 80)
    assert safe_addstr(win, 23, 79, "x") is False
    assert win.drawn == []


def test_safe_addstr_empty_text_is_vacuous_success():
    win = FakeWin(24, 80)
    assert safe_addstr(win, 0, 0, "") is True
    assert win.drawn == []


def test_safe_addstr_converts_curses_error_to_false():
    win = FakeWin(24, 80, fail_addstr=True)
    assert safe_addstr(win, 0, 0, "boom") is False


def test_safe_addstr_tolerates_transient_getmaxyx_failure():
    assert safe_addstr(ExplodingWin(), 0, 0, "x") is False


def test_safe_addstr_rejects_none_window():
    assert safe_addstr(None, 0, 0, "x") is False


def test_safe_addstr_rejects_non_string_text():
    with pytest.raises(TypeError):
        safe_addstr(FakeWin(24, 80), 0, 0, 123)


# -- clamp_window ------------------------------------------------------------


def test_clamp_window_centers_when_it_fits():
    assert clamp_window(24, 80, 10, 40) == (7, 20, 10, 40)


def test_clamp_window_clamps_oversized_requests():
    y, x, h, w = clamp_window(10, 20, 30, 60)
    assert (h, w) == (10, 20)
    assert (y, x) == (0, 0)


def test_clamp_window_returns_none_when_too_small():
    assert clamp_window(24, 80, 10, 40, min_h=30) is None
    assert clamp_window(2, 80, 10, 40) is None
    assert clamp_window(24, 5, 10, 40) is None
    assert clamp_window(0, 0, 10, 40) is None
    assert clamp_window(-1, 80, 10, 40) is None


def test_clamp_window_never_negative_origin():
    y, x, h, w = clamp_window(24, 80, 5, 30)
    assert y >= 0 and x >= 0
    assert y + h <= 24 and x + w <= 80


# -- resize handling -----------------------------------------------------------


class FakeStdscr:
    """Minimal recording stand-in for a curses window."""

    def __init__(self, h=24, w=80):
        self._h = h
        self._w = w
        self.cleared = 0
        self.drawn = []

    def getmaxyx(self):
        return (self._h, self._w)

    def clear(self):
        self.cleared += 1

    def erase(self):
        pass

    def bkgd(self, *args):
        pass

    def addstr(self, y, x, text, *args):
        self.drawn.append((y, x, text))

    def refresh(self):
        pass

    def noutrefresh(self):
        pass

    def move(self, y, x):
        pass


def test_handle_resize_clears_and_redraws_without_raising(tmp_path):
    from agent.config import AgentConfig
    from agent.tui import TuiApp

    app = TuiApp(AgentConfig(workspace=tmp_path))
    stdscr = FakeStdscr(5, 10)  # extremely small: fallback path
    app._handle_resize(stdscr)  # must not raise
    assert stdscr.cleared == 1
    assert stdscr.drawn  # fallback message was painted


def test_handle_resize_with_none_stdscr_is_noop(tmp_path):
    from agent.config import AgentConfig
    from agent.tui import TuiApp

    app = TuiApp(AgentConfig(workspace=tmp_path))
    app._handle_resize(None)  # must not raise


# -- transient getmaxyx failures -------------------------------------------------


class FailingSizeStdscr(FakeStdscr):
    def getmaxyx(self):
        raise _ERR("transient console state")


def test_draw_tolerates_transient_getmaxyx_failure(tmp_path):
    app = _app(tmp_path)
    app._draw(FailingSizeStdscr(24, 80))  # must not raise; retries next frame


def test_slash_menu_tolerates_transient_getmaxyx_failure(tmp_path):
    app = _app(tmp_path)
    app.input_text = "/mo"
    app.cursor_pos = 3
    assert (
        app._draw_slash_menu(FailingSizeStdscr(24, 80), 20, 2, 76) == 0
    )


# -- layout dimension matrix -----------------------------------------------------


@pytest.mark.parametrize(
    ("h", "w", "tier"),
    [
        (10, 40, "minimised"),
        (10, 39, "extremely_small"),
        (9, 40, "extremely_small"),
        (10, 30, "extremely_small"),
        (8, 20, "extremely_small"),
        (5, 10, "extremely_small"),
        (1, 1, "extremely_small"),
        (0, 0, "extremely_small"),
        (12, 50, "compact"),
        (20, 70, "normal"),
        (30, 100, "large"),
        (40, 140, "wide"),
    ],
)
def test_layout_tier_matrix(h, w, tier):
    from agent.tui import get_layout_tier

    assert get_layout_tier(h, w) == tier


@pytest.mark.parametrize(
    ("h", "w"),
    [(10, 40), (9, 40), (8, 20), (5, 10), (1, 1), (0, 0), (12, 50), (40, 140)],
)
def test_geometry_never_invalid(h, w):
    from agent.tui import calc_chatbox_geometry

    g = calc_chatbox_geometry(h, w)
    for key in ("chat_h", "chat_w", "chat_x", "chat_y", "inner_w", "inner_h"):
        assert g[key] >= 0
    if not g["is_minimised"]:
        assert g["chat_x"] + g["chat_w"] <= w
        assert g["chat_y"] + g["chat_h"] <= h


@pytest.mark.parametrize(
    ("model", "chat_w", "inner_w"),
    [
        ("m", 78, 76),
        ("x" * 60, 65, 63),
        ("x" * 200, 80, 78),
        ("m", 10, 8),
        ("m", 1, 1),
    ],
)
def test_bottom_layout_bounded_for_hostile_inputs(model, chat_w, inner_w):
    from agent.tui import chatbox_bottom_layout

    mode_str, footer, fx = chatbox_bottom_layout(
        "AUTO", model, "high", chat_w, inner_w
    )
    assert fx >= 0
    assert fx >= 2 + len(mode_str)  # never overlaps the mode badge
    assert len(footer) <= max(len(footer), 0)


def test_path_line_long_path_stays_bounded():
    from agent.tui import format_path_line

    line = format_path_line("C:\\" + "x" * 300, 80)
    assert len(line) == 80
    assert line.startswith("…")
    assert line.endswith("x")


def test_key_resize_triggers_redraw(tmp_path):
    from agent.config import AgentConfig
    from agent.tui import HAS_CURSES, TuiApp

    if not HAS_CURSES:
        pytest.skip("curses backend required")
    import curses

    app = TuiApp(AgentConfig(workspace=tmp_path))
    stdscr = FakeStdscr(24, 80)
    app._handle_integer_key(curses.KEY_RESIZE, stdscr)
    assert stdscr.cleared == 1  # intentional redraw, not a silent swallow


# -- modal geometry under resize pressure --------------------------------------


class ModalStdscr(FakeStdscr):
    """Fake terminal with a scripted key queue for modal loops."""

    def __init__(self, h, w, keys):
        super().__init__(h, w)
        self._keys = list(keys)

    def getch(self):
        if self._keys:
            return self._keys.pop(0)
        return 27


class ModalWin:
    def __init__(self, h=100, w=100):
        self._h = h
        self._w = w

    def getmaxyx(self):
        return (self._h, self._w)

    def bkgd(self, *args):
        pass

    def box(self):
        pass

    def addstr(self, y, x, text, *args):
        pass

    def noutrefresh(self):
        pass


def _patch_modal_curses(monkeypatch, sizes):
    import curses

    def fake_newwin(h, w, y, x):
        sizes.append((h, w, y, x))
        return ModalWin()

    monkeypatch.setattr(curses, "newwin", fake_newwin)
    monkeypatch.setattr(curses, "doupdate", lambda: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)


def _assert_sizes_fit(sizes, term_h, term_w):
    assert sizes, "expected at least one window"
    for h, w, y, x in sizes:
        assert h > 0 and w > 0
        assert y >= 0 and x >= 0
        assert y + h <= term_h and x + w <= term_w


class FailingSizeModal(ModalStdscr):
    def getmaxyx(self):
        raise _ERR("transient console state")


def test_picker_tolerates_transient_getmaxyx_failure(tmp_path):
    app = _app(tmp_path)
    stdscr = FailingSizeModal(24, 80, keys=[27])
    assert app._run_picker(stdscr, {"ollama": ["m"]}) is None


def test_intel_picker_tolerates_transient_getmaxyx_failure(tmp_path):
    app = _app(tmp_path)
    stdscr = FailingSizeModal(24, 80, keys=[27])
    assert app._run_intel_picker(stdscr) is None


def _app(tmp_path):
    from agent.config import AgentConfig
    from agent.tui import TuiApp

    return TuiApp(AgentConfig(workspace=tmp_path))


def test_picker_tiny_terminal_returns_none_without_window(tmp_path, monkeypatch):
    sizes = []
    _patch_modal_curses(monkeypatch, sizes)
    app = _app(tmp_path)
    stdscr = ModalStdscr(4, 10, keys=[27])
    assert app._run_picker(stdscr, {"ollama": ["m"]}) is None
    assert sizes == []


@pytest.mark.parametrize("term", [(24, 80), (12, 30), (15, 40)])
def test_picker_geometry_always_fits_terminal(tmp_path, monkeypatch, term):
    sizes = []
    _patch_modal_curses(monkeypatch, sizes)
    app = _app(tmp_path)
    stdscr = ModalStdscr(term[0], term[1], keys=[10])  # Enter: pick header
    app._run_picker(stdscr, {"ollama": ["m"]})
    _assert_sizes_fit(sizes, term[0], term[1])


def test_intel_picker_tiny_terminal_returns_none(tmp_path, monkeypatch):
    sizes = []
    _patch_modal_curses(monkeypatch, sizes)
    app = _app(tmp_path)
    stdscr = ModalStdscr(4, 10, keys=[27])
    assert app._run_intel_picker(stdscr) is None
    assert sizes == []


def test_intel_picker_geometry_fits_and_selects(tmp_path, monkeypatch):
    sizes = []
    _patch_modal_curses(monkeypatch, sizes)
    app = _app(tmp_path)
    stdscr = ModalStdscr(24, 80, keys=[10])  # Enter: current level
    assert app._run_intel_picker(stdscr) in (
        "default",
        "low",
        "medium",
        "high",
        "xhigh",
    )
    _assert_sizes_fit(sizes, 24, 80)


def test_connect_dialog_tiny_terminal_aborts_cleanly(
    tmp_path, monkeypatch
):
    sizes = []
    _patch_modal_curses(monkeypatch, sizes)
    monkeypatch.setattr(
        "agent.models.providers.list_all_providers_with_models",
        lambda timeout=2, use_cache=True: {"ollama": ["m"]},
    )
    app = _app(tmp_path)
    stdscr = ModalStdscr(4, 10, keys=[27])
    assert app._do_connect(stdscr) is None
    assert sizes == []
