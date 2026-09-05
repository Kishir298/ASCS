"""End-to-end coverage for the A.S.C.S. experience/learning system.

Covers: the persistent JSONL store (save/load/corruption/feedback/ranking/
contradiction penalties), configuration defaults, and the pipeline wiring that
makes verified prior outcomes influence future planning and context.
"""

from __future__ import annotations

import json
import time

from pytest import approx

from agent.config import AgentConfig
from agent.experience import Experience, ExperienceStore, format_for_prompt
from agent.loop import AgentLoop
from agent.planner import planner_prompt
from agent.prompts import system_prompt
from agent.workspace import Workspace


# ── helpers ─────────────────────────────────────────────────────────────────

class RecordingClient:
    """Scripted chat double that also records every messages batch it saw."""

    def __init__(self, responses, model="fake"):
        self.responses = list(responses)
        self.index = 0
        self.model = model
        self.captured: list[list[dict]] = []

    def chat(self, messages, *, format="json", options=None, timeout=None):
        self.captured.append(list(messages))
        if self.index < len(self.responses):
            item = self.responses[self.index]
            self.index += 1
            return item
        raise AssertionError("RecordingClient exhausted scripted responses")


def _plan_single_task(task_title="Write hello"):
    return json.dumps(
        {
            "tasks": [
                {
                    "id": "T1",
                    "title": task_title,
                    "kind": "implement",
                    "verification": ["report inspection findings"],
                }
            ]
        }
    )


def _looped_config(tmp_path, exp_path):
    return AgentConfig(
        workspace=tmp_path,
        mode="AUTO",
        experience_enabled=True,
        experience_path=str(exp_path),
    )


# ── store: persistence ──────────────────────────────────────────────────────

def test_store_save_load_roundtrip(tmp_path):
    store = ExperienceStore(path=tmp_path / "exp.jsonl")
    record = store.save_run(
        task="Add login flow",
        outcome="implemented with tests",
        success=True,
        plan="inspect -> edit -> test",
        actions=["write_file", "run_command"],
        observations=["pytest passed"],
        tags=["AUTO"],
    )
    loaded = store.load()
    assert len(loaded) == 1
    loaded_record = loaded[0]
    assert loaded_record.experience_id == record.experience_id
    assert loaded_record.task == "Add login flow"
    assert loaded_record.success
    assert loaded_record.score == 1.0  # default success score
    assert loaded_record.actions == ["write_file", "run_command"]


def test_store_save_run_failure_default_score(tmp_path):
    store = ExperienceStore(path=tmp_path / "exp.jsonl")
    store.save_run(task="Migrate DB", outcome="verification failed", success=False)
    record = store.load()[0]
    assert record.success is False
    assert record.score == -1.0  # default failure score


def test_store_skips_corrupt_records(tmp_path):
    path = tmp_path / "exp.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"task": "good one", "outcome": "ok", "success": true}',
                "this is not json {",
                '{"task": "", "outcome": "empty task must be dropped"}',
                '{"task": "second", "outcome": "fine", "success": false}',
                "",
            ]
        ),
        encoding="utf-8",
    )
    store = ExperienceStore(path=path)
    records = store.load()
    assert [r.task for r in records] == ["good one", "second"]


def test_store_update_feedback_rewrites_record(tmp_path):
    store = ExperienceStore(path=tmp_path / "exp.jsonl")
    record = store.save_run(task="Tidy imports", outcome="done", success=True)
    updated = store.update_feedback(
        record.experience_id,
        feedback="user corrected the approach",
        score=-0.6,
        success=False,
    )
    assert updated is not None
    assert updated.feedback == "user corrected the approach"
    assert updated.score == -0.6
    assert updated.success is False

    reloaded = store.load()[0]
    assert reloaded.feedback == "user corrected the approach"
    assert reloaded.score == -0.6

    # Unknown id returns None and leaves the store untouched.
    assert store.update_feedback("nope", feedback="x") is None
    assert len(store.load()) == 1


def test_store_search_ranks_relevance_over_recency(tmp_path):
    store = ExperienceStore(path=tmp_path / "exp.jsonl")
    # Old entry matching every query token.
    old = Experience(
        task="set up pytest and gitlab ci runner for every release",
        outcome="worked",
        success=True,
        timestamp=time.time() - 86400 * 60,
    )
    store.save(old)
    # Fresh entry matching only one query token.
    fresh = Experience(
        task="ci coverage report",
        outcome="works",
        success=True,
        timestamp=time.time(),
    )
    store.save(fresh)

    hits = store.search("pytest gitlab ci", limit=5)
    assert len(hits) == 2
    assert hits[0].experience_id == old.experience_id  # relevance wins


def test_store_search_successful_only_and_limit(tmp_path):
    store = ExperienceStore(path=tmp_path / "exp.jsonl")
    for i in range(4):
        store.save_run(
            task=f"refactor module {i}",
            outcome="ok",
            success=i % 2 == 0,
        )
    all_hits = store.search("refactor module", limit=5)
    assert len(all_hits) == 4
    success_hits = store.search("refactor module", limit=5, successful_only=True)
    assert len(success_hits) == 2
    assert all(h.success for h in success_hits)
    limited = store.search("refactor module", limit=2)
    assert len(limited) == 2


def test_store_penalize_contradictions(tmp_path):
    store = ExperienceStore(path=tmp_path / "exp.jsonl")
    store.save_run(task="Wire up the Ollama client", outcome="worked", success=True)
    original = store.load()[0]
    assert original.score == 1.0

    touched = store.penalize_contradictions("Wire up the Ollama client")
    assert len(touched) == 1
    assert store.load()[0].score < original.score

    # Repeated contradiction keeps degrading the same record.
    store.penalize_contradictions("Wire up the Ollama client")
    store.penalize_contradictions("Wire up the Ollama client")
    assert store.load()[0].score == approx(max(-1.0, original.score - 0.6))


def test_format_for_prompt_is_bounded(tmp_path):
    store = ExperienceStore(path=tmp_path / "exp.jsonl")
    for i in range(10):
        store.save_run(
            task=f"Long task {i} " + "word " * 200,
            outcome="ok",
            success=True,
        )
    text = format_for_prompt(store.load(), max_chars=600)
    assert len(text) <= 700  # bounded (one trailing newline over the strict cap)
    assert text
    assert "SUCCESS" in text


# ── config defaults ─────────────────────────────────────────────────────────

def test_config_keep_alive_defaults_to_30m(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_KEEP_ALIVE", raising=False)
    cfg = AgentConfig(workspace=tmp_path)
    assert cfg.keep_alive == "30m"


def test_experience_policy_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_EXPERIENCE_ENABLED", raising=False)
    monkeypatch.delenv("AGENT_EXPERIENCE_PATH", raising=False)
    from agent.config import load_config

    cfg = load_config(workspace=str(tmp_path))
    assert cfg.experience_enabled is True  # CLI pipeline learns by default
    assert cfg.experience_path is None
    # The library-layer default is off so unit tests stay hermetic.
    raw = AgentConfig(workspace=tmp_path)
    assert raw.experience_enabled is False


def test_experience_policy_env_overrides(monkeypatch, tmp_path):
    from agent.config import load_config

    monkeypatch.setenv("AGENT_EXPERIENCE_ENABLED", "0")
    monkeypatch.setenv("AGENT_EXPERIENCE_PATH", "C:/custom/ascs/exp.jsonl")
    cfg = load_config(workspace=str(tmp_path))
    assert cfg.experience_enabled is False
    assert cfg.experience_path == "C:/custom/ascs/exp.jsonl"


# ── prompt / planner injection ──────────────────────────────────────────────

def test_system_prompt_injects_experience_block(config):
    text = system_prompt(
        config,
        experience="Experience 1 [SUCCESS]\nTask: wire ollama\nOutcome: passed",
    )
    assert "PAST EXPERIENCE" in text
    assert "wire ollama" in text


def test_system_prompt_omits_experience_when_absent(config):
    text = system_prompt(config)
    assert "PAST EXPERIENCE" not in text


def test_planner_prompt_injects_experience_context():
    prompt = planner_prompt(
        "Add auth",
        "project info",
        experience_context="EXP-BLOCK-CONTENT",
    )
    assert "PAST EXPERIENCE" in prompt
    assert "EXP-BLOCK-CONTENT" in prompt


def test_planner_prompt_omits_experience_when_absent():
    prompt = planner_prompt("Add auth", "project info")
    assert "PAST EXPERIENCE" not in prompt


# ── pipeline: single-shot loop ──────────────────────────────────────────────

def test_run_records_experience_on_complete_not_cancel(tmp_path):
    exp_path = tmp_path / "exp.jsonl"
    config = _looped_config(tmp_path, exp_path)
    ws = Workspace(tmp_path)

    client = RecordingClient([json.dumps({"done": True, "summary": "all done"})])
    loop = AgentLoop(config, client, ws, log=lambda m: None)
    result = loop.run("Finish the widget")
    assert result.status == "completed"

    store = ExperienceStore(path=exp_path)
    records = store.load()
    assert len(records) == 1
    assert records[0].success
    assert "Finish the widget" in records[0].task
    assert "all done" in records[0].outcome

    # A cancelled run must not record anything new.
    cancelled_client = RecordingClient([json.dumps({"done": True, "summary": "x"})])
    cancelled_loop = AgentLoop(
        config,
        cancelled_client,
        ws,
        log=lambda m: None,
        should_stop=lambda: True,
    )
    cancelled_result = cancelled_loop.run("Another task")
    assert cancelled_result.status == "cancelled"
    assert len(store.load()) == 1
    assert cancelled_client.captured == []  # stopped before any model call


# ── pipeline: task-graph loop ───────────────────────────────────────────────

def test_run_graph_records_experience_after_completion(tmp_path):
    exp_path = tmp_path / "exp.jsonl"
    config = _looped_config(tmp_path, exp_path)
    client = RecordingClient(
        [
            _plan_single_task(),
            json.dumps({"done": True, "summary": "implemented hello"}),
        ]
    )
    ws = Workspace(tmp_path)
    loop = AgentLoop(config, client, ws, log=lambda m: None)
    result = loop.run_graph("Write hello")
    assert result.status == "completed"

    store = ExperienceStore(path=exp_path)
    records = store.load()
    assert len(records) == 1
    assert records[0].success
    assert "Write hello" in records[0].task


def test_experience_influences_next_plan(tmp_path):
    """A prior verified success must show up in the next planner prompt."""
    exp_path = tmp_path / "exp.jsonl"
    ExperienceStore(path=exp_path).save_run(
        task="Write hello",
        outcome="wrote hello with verification passed",
        success=True,
        tags=["AUTO"],
    )

    config = _looped_config(tmp_path, exp_path)
    client = RecordingClient(
        [
            _plan_single_task(),
            json.dumps({"done": True, "summary": "done"}),
        ]
    )
    ws = Workspace(tmp_path)
    loop = AgentLoop(config, client, ws, log=lambda m: None)
    result = loop.run_graph("Write hello")
    assert result.status == "completed"

    # The planner request is the first captured message batch.
    planner_text = client.captured[0][0]["content"]
    assert "PAST EXPERIENCE" in planner_text
    assert "wrote hello" in planner_text


def test_run_graph_records_failure_and_penalizes_contradiction(tmp_path):
    exp_path = tmp_path / "exp.jsonl"
    store = ExperienceStore(path=exp_path)
    store.save_run(task="Wire up the Ollama client", outcome="worked", success=True)
    prior = store.load()[0]

    failing_plan = json.dumps(
        {
            "tasks": [
                {
                    "id": "T1",
                    "title": "Wire up the Ollama client",
                    "kind": "implement",
                    "verification": ['run python -c "import sys; sys.exit(1)"'],
                }
            ]
        }
    )
    config = AgentConfig(
        workspace=tmp_path,
        mode="AUTO",
        experience_enabled=True,
        experience_path=str(exp_path),
        max_verify_retries=0,
    )
    client = RecordingClient(
        [
            failing_plan,
            json.dumps({"done": True, "summary": "implemented"}),
        ]
    )
    ws = Workspace(tmp_path)
    loop = AgentLoop(config, client, ws, log=lambda m: None)
    result = loop.run_graph("Wire up the Ollama client")
    assert result.status == "partial"

    records = store.load()
    failed = [r for r in records if not r.success]
    assert len(failed) == 1
    assert any("verification" in (r.errors or [""])[0].lower() for r in failed)
    # The earlier success was penalised by the contradiction.
    after = store.load()
    prior_after = next(r for r in after if r.experience_id == prior.experience_id)
    assert prior_after.score < prior.score