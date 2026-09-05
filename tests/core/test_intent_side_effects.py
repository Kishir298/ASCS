"""Loop-level intent-discipline regression tests (Phase 1).

The most important acceptance criterion of Phase 1: a request classified as
conversation/question/read-only inspection can NOT accidentally enter a write
path — even when the model insists on calling mutating tools.

These tests use a deliberately *hostile* fake client that always answers with
a ``write_file`` call. If any test in this module fails, the brain has leaked
side effects into a request that never authorized them.

The inverse invariant is also pinned: terse but legitimate work orders
(ambiguous intent) must still be executable — the gate must not over-block.
"""

from __future__ import annotations

import json

from agent.config import AgentConfig
from agent.loop import AgentLoop
from agent.workspace import Workspace


def tool_call(tool, arguments, comment="step"):
    return json.dumps({"comment": comment, "tool": tool, "arguments": arguments})


def done(summary):
    return json.dumps({"done": True, "summary": summary})


class HostileClient:
    """A model that always tries to create ``hello.txt``, no matter what.

    ``script`` (optional) supplies ordered one-shot responses that are
    consumed first; after the script is exhausted (or immediately, if empty),
    every reply is a ``write_file hello.txt`` call.
    """

    def __init__(self, script=None, model="fake-model"):
        self.script = list(script or [])
        self.index = 0
        self.model = model
        self.calls = []

    def chat(self, messages, *, format="json", options=None, timeout=None):
        self.calls.append(messages)
        if self.index < len(self.script):
            item = self.script[self.index]
            self.index += 1
            if isinstance(item, BaseException):
                raise item
            return item
        return tool_call(
            "write_file",
            {"path": "hello.txt", "content": "hi"},
            comment="I will create hello.txt",
        )


def make_loop(tmp_path, client, config_overrides=None):
    cfg_kwargs = {"workspace": tmp_path, "mode": "AUTO"}
    if config_overrides:
        cfg_kwargs.update(config_overrides)
    config = AgentConfig(**cfg_kwargs)
    ws = Workspace(tmp_path)
    loop = AgentLoop(config, client, ws, log=lambda m: None)
    return loop


# ---------------------------------------------------------------------------
# Scenario 1/2: conversation and questions get ZERO side effects
# ---------------------------------------------------------------------------


def test_hello_with_hostile_model_creates_nothing(tmp_path):
    """'hello' + a model that insists on write_file → no file, clean finish."""
    client = HostileClient()  # every response is a write_file attempt
    loop = make_loop(tmp_path, client)
    result = loop.run("hello")
    assert result.is_complete
    assert result.state == "complete"
    assert not (tmp_path / "hello.txt").exists()
    assert list(tmp_path.iterdir()) == []  # workspace untouched
    assert "(no tools were used)" in result.summary.lower()


def test_hi_with_hostile_model_creates_nothing(tmp_path):
    client = HostileClient()
    loop = make_loop(tmp_path, client)
    result = loop.run("hi")
    assert result.is_complete
    assert list(tmp_path.iterdir()) == []


def test_thanks_with_hostile_model_creates_nothing(tmp_path):
    client = HostileClient()
    loop = make_loop(tmp_path, client)
    result = loop.run("thanks")
    assert result.is_complete
    assert list(tmp_path.iterdir()) == []


def test_what_can_you_do_with_hostile_model_creates_nothing(tmp_path):
    client = HostileClient()
    loop = make_loop(tmp_path, client)
    result = loop.run("what can you do?")
    assert result.is_complete
    assert list(tmp_path.iterdir()) == []


def test_world_knowledge_question_with_hostile_model_creates_nothing(tmp_path):
    client = HostileClient()
    loop = make_loop(tmp_path, client)
    result = loop.run("what is Python?")
    assert result.is_complete
    assert list(tmp_path.iterdir()) == []


def test_conversational_answer_becomes_summary(tmp_path):
    client = HostileClient(script=[done("Hello! How can I help?")])
    loop = make_loop(tmp_path, client)
    result = loop.run("hello")
    assert result.is_complete
    assert "Hello!" in result.summary
    assert list(tmp_path.iterdir()) == []


def test_conversational_compliant_reply_is_single_call(tmp_path):
    """Fast path: a compliant done-reply finishes in ONE model call, no tools."""
    client = HostileClient(script=[done("Hi there!")])
    loop = make_loop(tmp_path, client)
    result = loop.run("hello")
    assert result.is_complete
    assert len(client.calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_conversational_hostile_reply_costs_exactly_one_extra_call(tmp_path):
    """A hostile tool reply costs exactly one answer-extraction call, then done."""
    client = HostileClient()  # first reply: write_file attempt
    loop = make_loop(tmp_path, client)
    result = loop.run("hello")
    assert result.is_complete
    assert len(client.calls) == 2
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Scenario 3: inspection requests may read, must never write
# ---------------------------------------------------------------------------


def test_inspection_request_write_attempt_is_refused_then_read_allowed(tmp_path):
    """'show me the files' + hostile write → write refused, read-only allowed."""
    client = HostileClient(
        script=[
            tool_call("write_file", {"path": "hello.txt", "content": "hi"}),
            tool_call("list_directory", {"path": "."}, comment="listing instead"),
            done("The project contains these files"),
        ]
    )
    loop = make_loop(tmp_path, client)
    result = loop.run("show me the files in this project")
    assert result.is_complete
    assert not (tmp_path / "hello.txt").exists()  # the write NEVER happened
    # The read-only fallback call did execute (third script slot consumed =
    # done; the list_directory slot was consumed second, so 3 calls total).
    assert len(client.calls) == 3


def test_inspection_request_with_pure_hostile_model_is_bounded_and_safe(tmp_path):
    """A model that never complies is stopped after bounded refusals — and the
    workspace still shows zero side effects. No infinite gate loop."""
    client = HostileClient()  # write_file forever
    loop = make_loop(tmp_path, client)
    result = loop.run("where is the database configured?")
    assert not result.is_complete
    assert result.status == "fatal"  # bounded: repeated-violation stop
    # The write NEVER happened: the only entry is ASCS's own .ascs state dir
    # (project manifest), created by context scanning — not a model side effect.
    assert not (tmp_path / "hello.txt").exists()
    assert {p.name for p in tmp_path.iterdir()} <= {".ascs"}


def test_what_is_in_this_project_write_attempt_refused(tmp_path):
    client = HostileClient(
        script=[
            tool_call("write_file", {"path": "hello.txt", "content": "hi"}),
            done("Nothing — the workspace is empty."),
        ]
    )
    loop = make_loop(tmp_path, client)
    result = loop.run("what is in this project?")
    assert result.is_complete
    assert not (tmp_path / "hello.txt").exists()


# ---------------------------------------------------------------------------
# Task-engine gate: run_graph must refuse conversational objectives
# ---------------------------------------------------------------------------


def test_run_graph_hello_never_plans_or_executes(tmp_path):
    client = HostileClient()  # always tries to write
    loop = make_loop(tmp_path, client)
    result = loop.run_graph("hello")
    assert result.is_complete
    assert result.task_count == 0  # no plan, no task graph
    assert list(tmp_path.iterdir()) == []  # nothing created, no .ascs state
    assert "(no tools were used)" in result.summary.lower()


def test_run_graph_thanks_never_plans_or_executes(tmp_path):
    client = HostileClient()
    loop = make_loop(tmp_path, client)
    result = loop.run_graph("thanks")
    assert result.is_complete
    assert result.task_count == 0
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Inverse invariant: legitimate work orders still execute
# ---------------------------------------------------------------------------


def test_ambiguous_work_order_can_still_write(tmp_path):
    """Terse work orders (ambiguous intent) must remain executable."""
    client = HostileClient(
        script=[
            tool_call("write_file", {"path": "a.txt", "content": "A"}),
            done("created a.txt"),
        ]
    )
    loop = make_loop(tmp_path, client)
    result = loop.run("make a.txt")
    assert result.is_complete
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "A"


def test_coding_request_write_still_executes(tmp_path):
    client = HostileClient(
        script=[
            tool_call(
                "write_file",
                {"path": "calc.py", "content": "print('calc')"},
            ),
            done("created calc.py"),
        ]
    )
    loop = make_loop(tmp_path, client)
    result = loop.run("create a Python calculator")
    assert result.is_complete
    assert (tmp_path / "calc.py").exists()


# ---------------------------------------------------------------------------
# Experience discipline: experience never fires for no-work requests
# ---------------------------------------------------------------------------


class _SpyExperience:
    """Records experience-store interactions without touching disk."""

    def __init__(self):
        self.search_calls = 0
        self.save_calls = 0

    def search(self, task_text, limit=5):
        self.search_calls += 1
        return []

    def save_run(self, **kwargs):
        self.save_calls += 1

        class _Rec:
            experience_id = "spy-1"

        return _Rec()

    def penalize_contradictions(self, task, exclude_id=None):
        pass


def test_conversational_request_never_touches_experience(tmp_path):
    loop = make_loop(
        tmp_path,
        HostileClient(script=[done("Hi!")]),
        config_overrides={"experience_enabled": True},
    )
    spy = _SpyExperience()
    loop.experience = spy
    result = loop.run("hello")
    assert result.is_complete
    assert spy.search_calls == 0
    assert spy.save_calls == 0


def test_coding_request_still_uses_experience(tmp_path):
    loop = make_loop(
        tmp_path,
        HostileClient(
            script=[
                tool_call("write_file", {"path": "x.txt", "content": "x"}),
                done("wrote x.txt"),
            ]
        ),
        config_overrides={"experience_enabled": True},
    )
    spy = _SpyExperience()
    loop.experience = spy
    result = loop.run("create a file named example.py")
    assert result.is_complete
    assert spy.search_calls >= 1
    assert spy.save_calls >= 1
