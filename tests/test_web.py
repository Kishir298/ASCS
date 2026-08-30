"""Tests for the web UI server, task runner, and cancellation plumbing."""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import pytest

from agent.config import AgentConfig
from agent.events import AgentEvent
from agent.web import App, EventHub, TaskRunner, interrupt_thread
from agent.workspace import Workspace


def _wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class RichFakeClient:
    """Scripted chat client that also answers the web-status probes."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0
        self.model = "fake-model"
        self.calls = []
        self.keep_alive = None

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if self.index < len(self.responses):
            item = self.responses[self.index]
            self.index += 1
            return item
        raise AssertionError("RichFakeClient exhausted")

    def check_connectivity(self, timeout=None):
        raise ConnectionError("no ollama in tests")

    def list_models(self, timeout=None):
        return []

    def abort_current(self):
        pass


def _app(tmp_path, responses=None):
    cfg = AgentConfig(workspace=tmp_path, mode="AUTO")
    client = RichFakeClient(
        responses
        if responses is not None
        else ['{"done": true, "summary": "ok"}']
    )
    return App(cfg, client, Workspace(tmp_path))


# -- EventHub ------------------------------------------------------------


def test_event_hub_fanout_and_history():
    hub = EventHub()
    q = hub.subscribe()
    hub.publish(AgentEvent(type="status", message="RUNNING"))
    item = q.get(timeout=1)
    assert item["type"] == "status"
    assert len(hub.history()) == 1
    assert hub.history()[0]["message"] == "RUNNING"
    hub.clear()
    assert hub.history() == []
    hub.unsubscribe(q)


def test_event_hub_history_since():
    hub = EventHub()
    hub.publish(AgentEvent(type="status", message="a"))
    hub.publish(AgentEvent(type="status", message="b"))
    hist = hub.history(since=1)
    assert len(hist) == 1
    assert hist[0]["message"] == "b"


# -- interrupt_thread ----------------------------------------------------


def test_interrupt_thread_raises_keyboard_interrupt():
    caught = threading.Event()
    result = {}

    def target():
        try:
            while True:
                time.sleep(0.01)
        except KeyboardInterrupt:
            result["ki"] = True
            caught.set()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    time.sleep(0.1)
    _wait_until(lambda: t.is_alive() and t.ident is not None)
    assert interrupt_thread(t) is True
    assert caught.wait(5)
    assert result.get("ki")


def test_interrupt_thread_dead_thread_returns_false():
    done = threading.Event()

    def target():
        done.set()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    done.wait(5)
    t.join(timeout=5)
    assert interrupt_thread(t) is False


# -- TaskRunner ----------------------------------------------------------


def test_task_runner_runs_to_completion(tmp_path):
    app = _app(tmp_path, ['{"done": true, "summary": "ok"}'])
    assert app.runner.start("say hi") is True
    assert _wait_until(lambda: not app.runner.busy)
    assert app.runner.result is not None
    assert app.runner.result.status == "completed"
    assert app.runner.result.state == "complete"


def test_task_runner_rejects_second_task_while_busy(tmp_path):
    blocking = threading.Event()
    keep = {"busy": True}

    class BlockingClient:
        model = "fake"

        def chat(self, messages, **kwargs):
            while keep["busy"]:
                time.sleep(0.01)

        def abort_current(self):
            pass

    cfg = AgentConfig(workspace=tmp_path, mode="AUTO")
    app = App(cfg, BlockingClient(), Workspace(tmp_path))
    assert app.runner.start("task one") is True
    time.sleep(0.3)
    assert app.runner.busy
    assert app.runner.start("task two") is False
    keep["busy"] = False
    assert _wait_until(lambda: not app.runner.busy)


def test_task_runner_cancel_returns_cancelled(tmp_path):
    keep = {"busy": True}

    class BlockingClient:
        model = "fake"

        def chat(self, messages, **kwargs):
            try:
                while keep["busy"]:
                    time.sleep(0.01)
            except KeyboardInterrupt:
                keep["busy"] = False
                raise

        def abort_current(self):
            pass

    cfg = AgentConfig(workspace=tmp_path, mode="AUTO")
    app = App(cfg, BlockingClient(), Workspace(tmp_path))
    assert app.runner.start("task") is True
    time.sleep(0.3)
    assert app.runner.busy
    assert app.runner.cancel() is True
    assert _wait_until(lambda: not app.runner.busy)
    assert app.runner.result is not None
    assert app.runner.result.status == "cancelled"
    assert app.runner.result.state == "cancelled"


def test_task_runner_unknown_mode_rejected(tmp_path):
    app = _app(tmp_path)
    with pytest.raises(ValueError):
        app.runner.start("task", mode="BOGUS")


def test_task_runner_per_run_mode_config(tmp_path):
    app = _app(tmp_path, ['{"done": true, "summary": "ok"}'])
    assert app.runner.start("task", mode="auto") is True
    assert _wait_until(lambda: not app.runner.busy)
    assert app.runner.mode == "AUTO"
    assert app.runner.result.status == "completed"


# -- HTTP endpoints ------------------------------------------------------


def test_app_state_payload(tmp_path):
    app = _app(tmp_path)
    snap = app.state()
    assert snap["busy"] is False
    assert snap["state"] == "idle"
    assert snap["mode"] == "AUTO"
    assert snap["version"]


def test_app_status_handles_offline_ollama(tmp_path):
    app = _app(tmp_path)
    status = app.status()
    assert status["server"]
    assert status["ollama"]["reachable"] is False
    assert status["ollama"]["installed"] == []


def test_app_status_cached_within_ttl(tmp_path):
    probes = {"n": 0}

    class CountingClient:
        model = "fake-model"
        keep_alive = None

        def check_connectivity(self, timeout=None):
            probes["n"] += 1
            raise ConnectionError("offline")

        def list_models(self, timeout=None):
            return []

        def abort_current(self):
            pass

    cfg = AgentConfig(workspace=tmp_path, mode="AUTO")
    app = App(cfg, CountingClient(), Workspace(tmp_path))
    app.status()
    app.status()
    app.status()
    assert probes["n"] == 1  # live probe performed once per TTL window
    # Force a fresh probe after the window elapses.
    app._status_at = 0.0
    app.status()
    assert probes["n"] == 2


def test_app_warm_status_prepopulates_cache(tmp_path):
    probes = {"n": 0}

    class CountingClient:
        model = "fake-model"
        keep_alive = None

        def check_connectivity(self, timeout=None):
            probes["n"] += 1

        def list_models(self, timeout=None):
            return []

        def abort_current(self):
            pass

    cfg = AgentConfig(workspace=tmp_path, mode="AUTO")
    app = App(cfg, CountingClient(), Workspace(tmp_path))
    app.warm_status()
    assert probes["n"] == 1
    assert app.status()["ollama"]["reachable"] is True
    assert probes["n"] == 1  # served from cache, no second live probe
    app.status()
    app.status()
    assert probes["n"] == 1  # still cached inside the TTL window


def test_http_endpoints_smoke(tmp_path):
    app = _app(tmp_path, ['{"done": true, "summary": "web task done"}'])
    url = app.start()
    try:
        with urllib.request.urlopen(url + "/api/state", timeout=5) as r:
            assert r.status == 200
            body = json.loads(r.read().decode())
            assert body["state"] == "idle"

        with urllib.request.urlopen(url + "/api/status", timeout=5) as r:
            status = json.loads(r.read().decode())
            assert status["ollama"]["reachable"] is False

        with urllib.request.urlopen(url + "/", timeout=5) as r:
            html = r.read().decode()
            assert "A.S.C.S." in html

        req = urllib.request.Request(
            url + "/api/task",
            data=json.dumps({"task": "web task", "mode": "AUTO"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 202
            assert json.loads(r.read().decode())["started"] is True

        assert _wait_until(lambda: not app.runner.busy)
        with urllib.request.urlopen(url + "/api/state", timeout=5) as r:
            body = json.loads(r.read().decode())
            assert body["result"]["status"] == "completed"

        with urllib.request.urlopen(url + "/api/history", timeout=5) as r:
            data = json.loads(r.read().decode())
            assert data["events"]
    finally:
        app.stop()


def test_http_task_requires_text(tmp_path):
    app = _app(tmp_path)
    url = app.start()
    try:
        req = urllib.request.Request(
            url + "/api/task",
            data=json.dumps({"mode": "AUTO"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 400
    finally:
        app.stop()


def test_sse_streams_server_hello(tmp_path):
    app = _app(tmp_path)
    url = app.start()
    try:
        resp = urllib.request.urlopen(url + "/api/events", timeout=5)
        raw = resp.read(512)
        text = raw.decode("utf-8", errors="replace")
        assert "event: server" in text
        assert '"version"' in text
        resp.close()
    finally:
        app.stop()


def test_http_stop_cancels_active_run(tmp_path):
    keep = {"busy": True}

    class BlockingClient:
        model = "fake"

        def chat(self, messages, **kwargs):
            try:
                while keep["busy"]:
                    time.sleep(0.01)
            except KeyboardInterrupt:
                keep["busy"] = False
                raise

        def abort_current(self):
            pass

    cfg = AgentConfig(workspace=tmp_path, mode="AUTO")
    app = App(cfg, BlockingClient(), Workspace(tmp_path))
    url = app.start()
    try:
        req = urllib.request.Request(
            url + "/api/task",
            data=json.dumps({"task": "long"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5).read()
        time.sleep(0.3)
        assert app.runner.busy

        stop_req = urllib.request.Request(url + "/api/stop", method="POST")
        with urllib.request.urlopen(stop_req, timeout=5) as r:
            payload = json.loads(r.read().decode())
            assert payload["cancelling"] is True

        assert _wait_until(lambda: not app.runner.busy)
        assert app.runner.result.status == "cancelled"
    finally:
        app.stop()