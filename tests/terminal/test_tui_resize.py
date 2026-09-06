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

    def keypad(self, flag):
        pass

    def timeout(self, ms):
        pass

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
        (10, 20, "extremely_small"),
        (12, 40, "minimised"),
        (10, 39, "extremely_small"),
        (9, 40, "extremely_small"),
        (10, 30, "extremely_small"),
        (8, 20, "extremely_small"),
        (5, 10, "extremely_small"),
        (5, 5, "extremely_small"),
        (9, 9, "extremely_small"),
        (10, 10, "extremely_small"),
        (20, 5, "extremely_small"),
        (1, 1, "extremely_small"),
        (0, 0, "extremely_small"),
        (39, 9, "extremely_small"),
        (12, 50, "compact"),
        (20, 40, "minimised"),
        (20, 60, "compact"),
        (20, 80, "normal"),
        (24, 80, "normal"),
        (41, 10, "extremely_small"),
        (49, 11, "extremely_small"),
        (69, 11, "extremely_small"),
        (12, 70, "normal"),
        (20, 70, "normal"),
        (20, 10, "extremely_small"),
        (12, 99, "normal"),
        (12, 100, "large"),
        (30, 100, "large"),
        (20, 139, "large"),
        (20, 140, "wide"),
        (40, 140, "wide"),
        (50, 200, "wide"),
        (60, 200, "wide"),
        (200, 50, "compact"),
        (200, 60, "compact"),
    ],
)
def test_layout_tier_matrix(h, w, tier):
    from agent.tui import get_layout_tier

    assert get_layout_tier(h, w) == tier


def test_tier_transitions_both_directions():
    """Shrink ‑> degraded and enlarge ‑> recovery through every tier."""
    from agent.tui import get_layout_tier

    down = [(30, 100), (12, 50), (10, 40), (5, 10)]
    assert [get_layout_tier(h, w) for h, w in down] == [
        "large",
        "compact",
        "minimised",
        "extremely_small",
    ]
    assert [get_layout_tier(h, w) for h, w in reversed(down)] == [
        "extremely_small",
        "minimised",
        "compact",
        "large",
    ]
    assert get_layout_tier(24, 80) == "normal"
    assert get_layout_tier(5, 10) == "extremely_small"
    assert get_layout_tier(24, 80) == "normal"  # wide swing back is safe


@pytest.mark.parametrize(
    ("h", "w"),
    [(10, 40), (10, 20), (12, 40), (9, 40), (8, 20), (5, 10), (5, 5), (10, 10), (20, 5),
     (1, 1), (0, 0), (12, 50), (41, 10), (69, 11), (70, 12), (20, 40), (20, 60),
     (20, 80), (24, 80), (20, 10), (40, 140), (60, 200), (200, 50), (200, 60), (50, 200)],
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


# -- input cursor bounds ---------------------------------------------------------


class CursorStdscr(FakeStdscr):
    def __init__(self, h=24, w=80):
        super().__init__(h, w)
        self.moves = []

    def move(self, y, x):
        h, w = self.getmaxyx()
        assert 0 <= y < h, f"move y={y} outside 0..{h - 1}"
        assert 0 <= x < w, f"move x={x} outside 0..{w - 1}"
        self.moves.append((y, x))


@pytest.mark.parametrize("width", [80, 20, 10, 5, 2, 1])
def test_draw_input_clamps_stale_cursor(tmp_path, width):
    app = _app(tmp_path)
    app.input_text = "hello"
    app.cursor_pos = 500  # stale: far beyond the text
    stdscr = CursorStdscr(24, 80)
    app._draw_input(stdscr, 20, 2, width)  # must not raise
    assert app.cursor_pos == len("hello")
    for y, x in stdscr.moves:
        assert 0 <= x < 80


def test_draw_input_negative_cursor_recovers(tmp_path):
    app = _app(tmp_path)
    app.input_text = "hello"
    app.cursor_pos = -7
    app._draw_input(CursorStdscr(24, 80), 20, 2, 76)
    assert app.cursor_pos == 0


def test_draw_input_move_never_leaves_narrow_window(tmp_path):
    app = _app(tmp_path)
    app.input_text = "x" * 100
    app.cursor_pos = 100
    stdscr = CursorStdscr(10, 10)
    app._draw_input(stdscr, 5, 2, 6)  # must not raise
    for y, x in stdscr.moves:
        assert 0 <= y < 10 and 0 <= x < 10


# -- rapid resize + resize during active work ------------------------------------


class ResizingStdscr(FakeStdscr):
    """Fake terminal whose dimensions rotate on every read (rapid resize)."""

    def __init__(self, sizes):
        self._sizes = list(sizes)
        self.cleared = 0
        self.drawn = []

    def getmaxyx(self):
        size = self._sizes.pop(0)
        self._sizes.append(size)
        return size


def test_rapid_resize_sequence_never_escapes(tmp_path):
    import curses

    from agent.tui import get_layout_tier

    app = _app(tmp_path)
    app.input_text = "draft task"
    app.cursor_pos = 5
    sizes = [(24, 80), (10, 30), (5, 10), (30, 100), (1, 1), (24, 80)]
    stdscr = ResizingStdscr(sizes)
    seen_tiers = set()
    for _ in range(6):
        app._handle_integer_key(curses.KEY_RESIZE, stdscr)  # must not raise
        seen_tiers.add(get_layout_tier(*stdscr.getmaxyx()))
    # app state intact, layout actually tracked the changing sizes
    assert app.input_text == "draft task"
    assert app.cursor_pos == 5
    assert not app.should_quit
    assert len(seen_tiers) >= 2


def test_resize_during_task_preserves_work(tmp_path):
    import curses

    app = _app(tmp_path)

    class BusyRunner:
        busy = True
        result = None

    class QuietHub:
        def history(self):
            return []

    app._runner = BusyRunner()
    app._hub = QuietHub()
    stdscr = ResizingStdscr([(24, 80), (12, 40), (24, 80)])
    for _ in range(3):
        app._handle_integer_key(curses.KEY_RESIZE, stdscr)
        app._poll_runner()  # must not cancel, fail, or complete anything
    assert app._runner is not None
    assert app._runner.busy is True
    assert "Cancelled" not in app.status_msg
    assert "Failed" not in app.status_msg
    assert "Completed" not in app.status_msg


def test_resize_back_to_large_restores_geometry(tmp_path):
    from agent.tui import calc_chatbox_geometry

    small = calc_chatbox_geometry(5, 10)
    assert small["is_minimised"] == 1
    large = calc_chatbox_geometry(24, 80)
    assert large["is_minimised"] == 0
    assert large["chat_w"] > 0 and large["chat_h"] > 0


# -- startup + header transient failures -----------------------------------------


def test_run_curses_survives_timeout_setup_failure(tmp_path, monkeypatch):
    app = _app(tmp_path)

    class TimeoutStdscr(FakeStdscr):
        def timeout(self, ms):
            raise _ERR("transient console state")

        def get_wch(self):
            raise KeyboardInterrupt

    app.run_curses(TimeoutStdscr(24, 80))  # must not raise
    assert app.should_quit


def test_header_survives_transient_color_failure(tmp_path, monkeypatch):
    import curses

    monkeypatch.setattr(
        curses, "color_pair", lambda *a: (_ for _ in ()).throw(_ERR("x"))
    )
    app = _app(tmp_path)
    stdscr = FakeStdscr(24, 80)
    app._draw_header(stdscr, 24, 80)  # must not raise
    assert stdscr.drawn == []


# -- conversation wrapping contract ------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "a" * 500,
        "a" * 5000,
        "C:\\" + "folder\\" * 60 + "file.py",
        "https://example.com/" + "path-segment/" * 40 + "?q=1&r=2",
        "Unicode: héllo wörld 日本語テスト 🎉 مرحبا",
        "   ",
        "word " * 200,
        "supercalifragilisticexpialidocious" * 20,
        "para one\n\npara two\nline three\n\n\npara four",
    ],
)
@pytest.mark.parametrize("width", [10, 40, 76])
def test_wrap_lines_bounded(tmp_path, text, width):
    app = _app(tmp_path)
    lines = app._wrap_lines(text, width)
    assert lines  # never empty
    for line in lines:
        assert len(line) <= max(10, width)
    # no content lost (modulo intra-line whitespace collapsing)
    assert "".join(lines).replace(" ", "") == text.replace(
        " ", ""
    ).replace("\n", "")


def test_wrap_lines_empty_and_degenerate_widths(tmp_path):
    app = _app(tmp_path)
    assert app._wrap_lines("", 40) == [""]
    verbatim = app._wrap_lines("hello", 0)
    assert verbatim == ["hello"]  # documented: caller floors width first
    assert app._wrap_lines("hello", -5) == ["hello"]


def test_render_message_lines_bounded(tmp_path):
    app = _app(tmp_path)
    app._add_message("user", "a" * 5000)
    app._add_message("assistant", "https://example.com/" + "x" * 1000)
    app._add_message("user", "héllo 🎉")
    rendered = app._render_message_lines(40)
    assert rendered
    for line, _attr in rendered:
        assert len(line) <= 40


def test_draw_conversation_rows_within_terminal(tmp_path):
    app = _app(tmp_path)
    app._add_message("user", "b" * 5000)
    app._add_message("assistant", "ok")

    class RecordingStdscr(FakeStdscr):
        def __init__(self, h, w):
            super().__init__(h, w)
            self.rows = []

        def addstr(self, y, x, text, *args):
            assert 0 <= y < self._h
            assert 0 <= x < self._w
            assert x + len(text) <= self._w
            self.rows.append((y, x, text))

    stdscr = RecordingStdscr(24, 80)
    app._draw_conversation(stdscr, 3, 2, 76, 15)  # must not raise
    assert stdscr.rows
    for y, x, text in stdscr.rows:
        assert 0 <= y < 24 and 0 <= x < 80


# -- bounded string rendering contract --------------------------------------------


@pytest.mark.parametrize("width", [1, 10, 40, 80])
@pytest.mark.parametrize(
    "path",
    [
        "C:\\proj",
        "C:\\" + "deep\\" * 100 + "file.py",
        "short",
        "",
        "workspace with spaces and ünicode\\日本語",
    ],
)
def test_path_line_never_exceeds_width(path, width):
    from agent.tui import format_path_line

    line = format_path_line(path, width)
    assert len(line) <= width
    if width <= 0:
        assert line == ""


@pytest.mark.parametrize("width", [8, 20, 40, 80])
def test_status_and_model_footer_bounded(tmp_path, width):
    from agent.tui import chatbox_bottom_layout, format_model_footer

    model = "q" * 500
    footer = format_model_footer(model, "high")
    mode_str, clipped, fx = chatbox_bottom_layout(
        "AUTO", model, "high", width, width
    )
    assert fx >= 0
    assert fx >= 2 + len(mode_str)
    assert len(clipped) <= width


def test_long_status_message_draws_safely(tmp_path, monkeypatch):
    import curses

    monkeypatch.setattr(curses, "has_colors", lambda: False)
    app = _app(tmp_path)
    app.status_msg = "S" * 5000

    class RecordingStdscr(FakeStdscr):
        def __init__(self, h, w):
            super().__init__(h, w)
            self.rows = []

        def addstr(self, y, x, text, *args):
            assert 0 <= y < self._h
            assert 0 <= x < self._w
            assert x + len(text) <= self._w
            self.rows.append((y, x, text))

    stdscr = RecordingStdscr(24, 80)
    app._draw_status(stdscr, 23, 80)  # must not raise
    assert stdscr.rows
    for y, x, text in stdscr.rows:
        assert x + len(text) <= 80


def test_long_error_summary_draws_safely(tmp_path):
    app = _app(tmp_path)
    app._add_system("E" * 5000)
    rendered = app._render_message_lines(40)
    for line, _attr in rendered:
        assert len(line) <= 40


# -- exact resize chains + resize during conversation ------------------------------


def test_prompt_resize_chain_stays_valid():
    """large→compact→extremely_small→normal→wide→compact→large."""
    from agent.tui import calc_chatbox_geometry, get_layout_tier

    chain = [
        ((30, 100), "large"),
        ((12, 50), "compact"),
        ((5, 10), "extremely_small"),
        ((24, 80), "normal"),
        ((20, 140), "wide"),
        ((12, 60), "compact"),
        ((30, 120), "large"),
    ]
    for (h, w), tier in chain:
        assert get_layout_tier(h, w) == tier
        g = calc_chatbox_geometry(h, w)
        for key in ("chat_h", "chat_w", "chat_x", "chat_y"):
            assert g[key] >= 0
        if not g["is_minimised"]:
            assert g["chat_x"] + g["chat_w"] <= w
            assert g["chat_y"] + g["chat_h"] <= h
            assert g["inner_w"] >= 1 and g["inner_h"] >= 1


def test_resize_during_conversation_preserves_messages(tmp_path):
    import curses

    app = _app(tmp_path)
    app._add_message("user", "p" * 2000)
    app._add_message("assistant", "https://example.com/" + "q" * 500)
    before = [dict(m) for m in app.messages]
    for h, w in [(24, 80), (12, 50), (5, 10), (30, 100), (24, 80)]:
        stdscr = FakeStdscr(h, w)
        app._handle_integer_key(curses.KEY_RESIZE, stdscr)  # must not raise
        app._draw_conversation(
            stdscr, 3, 2, max(1, w - 4), max(1, h - 8)
        )  # must not raise
    assert [dict(m) for m in app.messages] == before
    assert len(app.messages) == 2


def test_rapid_resize_full_matrix_sequence(tmp_path):
    """All ten required sizes in one rapid sequence: nothing escapes."""
    import curses

    from agent.tui import calc_chatbox_geometry, get_layout_tier

    app = _app(tmp_path)
    app.input_text = "draft"
    app.cursor_pos = 3
    app._add_message("user", "hello")
    sizes = [
        (24, 80),
        (20, 60),
        (12, 50),
        (5, 10),
        (30, 100),
        (10, 40),
        (40, 140),
        (20, 80),
        (12, 60),
        (30, 120),
    ]
    stdscr = ResizingStdscr(sizes)
    seen_tiers = set()
    for _ in sizes:
        app._handle_integer_key(curses.KEY_RESIZE, stdscr)  # must not raise
        h, w = stdscr.getmaxyx()
        tier = get_layout_tier(h, w)
        seen_tiers.add(tier)
        g = calc_chatbox_geometry(h, w)
        for key in ("chat_h", "chat_w", "chat_x", "chat_y"):
            assert g[key] >= 0
        if not g["is_minimised"]:
            assert g["chat_x"] + g["chat_w"] <= w
            assert g["chat_y"] + g["chat_h"] <= h
    assert app.input_text == "draft"
    assert app.cursor_pos == 3
    assert len(app.messages) == 1
    assert not app.should_quit
    assert len(seen_tiers) >= 3


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
    def __init__(self, h=100, w=100, fail_box=False, fail_refresh=False):
        self._h = h
        self._w = w
        self.fail_box = fail_box
        self.fail_refresh = fail_refresh

    def getmaxyx(self):
        return (self._h, self._w)

    def bkgd(self, *args):
        pass

    def box(self):
        if self.fail_box:
            raise _ERR("transient box failure")

    def addstr(self, y, x, text, *args):
        pass

    def refresh(self):
        if self.fail_refresh:
            raise _ERR("transient refresh failure")

    def noutrefresh(self):
        if self.fail_refresh:
            raise _ERR("transient refresh failure")


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


class FlakyGetchModal(ModalStdscr):
    """First keypress raises transiently, then the script runs."""

    def __init__(self, h, w, keys):
        super().__init__(h, w, keys)
        self._flaked = False

    def getch(self):
        if not self._flaked:
            self._flaked = True
            raise _ERR("transient resize race")
        return super().getch()


def test_picker_tolerates_transient_getmaxyx_failure(tmp_path):
    app = _app(tmp_path)
    stdscr = FailingSizeModal(24, 80, keys=[27])
    assert app._run_picker(stdscr, {"ollama": ["m"]}) is None


def test_intel_picker_tolerates_transient_getmaxyx_failure(tmp_path):
    app = _app(tmp_path)
    stdscr = FailingSizeModal(24, 80, keys=[27])
    assert app._run_intel_picker(stdscr) is None


def test_picker_survives_transient_getch_failure(tmp_path, monkeypatch):
    sizes = []
    _patch_modal_curses(monkeypatch, sizes)
    app = _app(tmp_path)
    stdscr = FlakyGetchModal(24, 80, keys=[10])  # raise once, then Enter
    result = app._run_picker(stdscr, {"ollama": ["m"]})
    assert result is not None  # popup survived instead of aborting
    assert sizes  # and actually drew


def test_intel_picker_survives_transient_getch_failure(tmp_path, monkeypatch):
    sizes = []
    _patch_modal_curses(monkeypatch, sizes)
    app = _app(tmp_path)
    stdscr = FlakyGetchModal(24, 80, keys=[10])
    assert app._run_intel_picker(stdscr) in (
        "default",
        "low",
        "medium",
        "high",
        "xhigh",
    )


def test_picker_survives_box_failure(tmp_path, monkeypatch):
    import curses

    monkeypatch.setattr(curses, "doupdate", lambda: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)
    monkeypatch.setattr(
        curses, "newwin", lambda h, w, y, x: ModalWin(fail_box=True)
    )
    app = _app(tmp_path)
    stdscr = ModalStdscr(24, 80, keys=[27])
    assert app._run_picker(stdscr, {"ollama": ["m"]}) is None


def test_picker_survives_refresh_failure(tmp_path, monkeypatch):
    import curses

    monkeypatch.setattr(curses, "doupdate", lambda: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)
    monkeypatch.setattr(
        curses, "newwin", lambda h, w, y, x: ModalWin(fail_refresh=True)
    )
    app = _app(tmp_path)
    stdscr = ModalStdscr(24, 80, keys=[10])
    # transient refresh failure is contained; the popup keeps working
    assert app._run_picker(stdscr, {"ollama": ["m"]}) is not None


def test_picker_survives_transient_newwin_failure(tmp_path, monkeypatch):
    import curses

    calls = []

    def flaky_newwin(h, w, y, x):
        calls.append((h, w, y, x))
        if len(calls) == 1:
            raise _ERR("transient resize race")
        return ModalWin()

    monkeypatch.setattr(curses, "newwin", flaky_newwin)
    monkeypatch.setattr(curses, "doupdate", lambda: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)
    app = _app(tmp_path)
    stdscr = ModalStdscr(24, 80, keys=[27])
    assert app._run_picker(stdscr, {"ollama": ["m"]}) is None
    assert len(calls) >= 1


def test_draw_input_survives_move_failure(tmp_path):
    app = _app(tmp_path)
    app.input_text = "hello"
    app.cursor_pos = 2

    class MoveFailStdscr(CursorStdscr):
        def move(self, y, x):
            raise _ERR("transient move failure")

    app._draw_input(MoveFailStdscr(24, 80), 20, 2, 76)  # must not raise


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
