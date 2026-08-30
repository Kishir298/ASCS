"""Local web UI server for A.S.C.S.

A standard-library-only HTTP server (no framework, no frontend build) that:

    * serves a single-page UI at ``GET /`` (``agent/ui/index.html``),
    * accepts tasks via ``POST /api/task``,
    * streams live agent events to browsers via Server-Sent Events at
      ``GET /api/events``,
    * exposes state/history/status via ``GET /api/state|history|status``,
    * cancels the active run via ``POST /api/stop`` (and ``POST /api/clear``
      clears history).

The agent runs in a dedicated worker thread so the HTTP server never blocks
on model/tool execution. Cancellation interrupts the worker thread (raising
``KeyboardInterrupt`` at the Python level) and aborts any in-flight Ollama
request by closing its socket, then kills any child process trees so no
orphaned subprocesses are left behind.
"""

from __future__ import annotations

import ctypes
import json
import queue
import threading
import time as _time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __version__
from .config import AgentConfig, MODES
from .events import AgentEvent
from .loop import AgentLoop, LoopResult
from .state import IDLE, STATE_LABELS, StateTracker
from .workspace import Workspace

UI_DIR = Path(__file__).resolve().parent / "ui"
INDEX_HTML: str | None = None


def _load_index() -> str:
    global INDEX_HTML
    if INDEX_HTML is None:
        path = UI_DIR / "index.html"
        if path.exists():
            INDEX_HTML = path.read_text(encoding="utf-8")
        else:
            INDEX_HTML = _fallback_html()
    return INDEX_HTML


def _fallback_html() -> str:
    return (
        "<!doctype html><meta charset=utf-8><title>A.S.C.S.</title>"
        "<h1>A.S.C.S.</h1><p>UI asset not found (agent/ui/index.html missing).</p>"
    )


class EventHub:
    """Fan-out of structured events to SSE subscribers with retained history."""

    def __init__(self, max_history: int = 5000) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._history: list[dict[str, Any]] = []
        self._max_history = max_history

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, event: AgentEvent) -> None:
        payload = event.to_dict()
        with self._lock:
            self._history.append(payload)
            if len(self._history) > self._max_history:
                del self._history[: len(self._history) - self._max_history]
            for sub in list(self._subscribers):
                try:
                    sub.put_nowait(payload)
                except queue.Full:
                    try:
                        sub.get_nowait()
                        sub.put_nowait(payload)
                    except queue.Empty:  # pragma: no cover - defensive
                        pass

    def history(self, since: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history[since:])

    def clear(self) -> None:
        with self._lock:
            self._history.clear()


def interrupt_thread(thread: threading.Thread) -> bool:
    """Raise KeyboardInterrupt inside ``thread`` if it is alive and Pythonic.

    Returns True when the interrupt was scheduled. Uses the classic
    ``PyThreadState_SetAsyncExc`` trick (also used by debuggers/Jupyter).
    """
    if not thread.is_alive() or thread.ident is None:
        return False
    tid = ctypes.c_ulong(thread.ident)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        tid, ctypes.py_object(KeyboardInterrupt)
    )
    if res > 1:  # pragma: no cover - defensive; unschedule any stray exc
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        return False
    return res == 1


class TaskRunner:
    """Runs one agent loop in a background thread with real cancellation."""

    def __init__(
        self, config: AgentConfig, client: Any, workspace: Workspace, hub: EventHub
    ) -> None:
        self._config = config
        self._client = client
        self._ws = workspace
        self._hub = hub
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._result: LoopResult | None = None
        self._mode: str = config.mode
        self._task: str = ""
        self.tracker = StateTracker(IDLE)

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def result(self) -> LoopResult | None:
        with self._lock:
            return self._result

    @property
    def mode(self) -> str:
        return self._mode

    def start(self, task: str, mode: str | None = None) -> bool:
        if mode:
            mode = mode.upper()
            if mode not in (*MODES, "SAFE"):
                raise ValueError(f"Unknown mode {mode!r}")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self._result = None
            self._thread = threading.Thread(
                target=self._work,
                args=(task, mode),
                name=f"ascs-run-{_time.time():.0f}",
                daemon=True,
            )
            self._thread.start()
        return True

    def _work(self, task: str, mode: str | None) -> None:
        run_config = self._config
        if mode is not None and mode.upper() != self._config.mode.upper():
            run_config = replace(self._config, mode=mode.upper())
        self._mode = run_config.mode
        self._task = task
        self.tracker.configure(mode=run_config.mode, task=task)

        def should_stop() -> bool:
            return self._stop.is_set()

        loop = AgentLoop(
            run_config,
            self._client,
            self._ws,
            log=lambda m: self._hub.publish(
                AgentEvent(type="activity", message=m, status="log")
            ),
            event_sink=self._hub.publish,
            should_stop=should_stop,
            tracker=self.tracker,
        )
        try:
            result = loop.run(task)
        except Exception as exc:  # never let a worker crash silently
            result = LoopResult(
                status="fatal",
                state="failed",
                summary=f"Worker crash: {exc}",
                error=str(exc),
            )
        with self._lock:
            self._result = result

    def cancel(self) -> bool:
        was_busy = self.busy
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            interrupt_thread(thread)
            try:
                self._client.abort_current()
            except Exception:  # pragma: no cover - defensive
                pass
        return was_busy


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"A.S.C.S/{__version__}"

    # -- helpers ------------------------------------------------------------

    @property
    def app(self) -> "App":
        return self.server.app  # type: ignore[attr-defined]

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            return {}
        if length <= 0:
            return {}
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _send_asset(self) -> None:
        html = _load_index().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    # -- routing ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/":
            self._send_asset()
            return
        if path == "/api/state":
            self._send_json(200, self.app.state())
            return
        if path == "/api/history":
            self._send_json(200, {"events": self.app.hub.history()})
            return
        if path == "/api/status":
            self._send_json(200, self.app.status())
            return
        if path == "/api/events":
            self._stream_events()
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        path = self.path.split("?")[0].rstrip("/")
        if path == "/api/task":
            data = self._read_json()
            task = (data.get("task") or "").strip()
            if not task:
                self._send_json(400, {"error": "task text is required"})
                return
            mode = data.get("mode") or self.app.config.mode
            try:
                started = self.app.runner.start(task, mode)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            if not started:
                self._send_json(409, {"error": "an agent run is already active"})
                return
            self._send_json(202, {"started": True, "mode": self.app.runner.mode})
            return
        if path == "/api/stop":
            was_busy = self.app.runner.cancel()
            self._send_json(200, {"cancelling": was_busy, "busy": self.app.runner.busy})
            return
        if path == "/api/clear":
            self.app.hub.clear()
            self.app.runner.tracker.reset()
            self._send_json(200, {"cleared": True})
            return
        self._send_json(404, {"error": "not found"})

    # -- SSE ----------------------------------------------------------------

    def _stream_events(self) -> None:
        app = self.app
        sub = app.hub.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            hello = {
                "type": "server",
                "version": __version__,
                "state": app.state(),
                "history_since": len(app.hub.history()),
            }
            self._sse_write("server", hello)

            while not app._shutdown:
                try:
                    item = sub.get(timeout=15)
                    self._sse_write("event", item)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    try:
                        self.wfile.flush()
                    except OSError:
                        break
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client went away
        finally:
            app.hub.unsubscribe(sub)
            try:
                self.close_connection = True
            except Exception:  # pragma: no cover - defensive
                pass

    def _sse_write(self, event_name: str, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"event: {event_name}\ndata: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def log_message(self, fmt: str, *args: Any) -> None:  # keep console quiet-ish
        print(f"[web] {self.client_address[0]} {fmt % args}")


class ASCSHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int], app: "App") -> None:
        super().__init__(addr, _Handler)
        self.app = app


class App:
    """State + runner + hub behind the HTTP server."""

    def __init__(self, config: AgentConfig, client: Any, workspace: Workspace) -> None:
        self.config = config
        self.client = client
        self.ws = workspace
        self.hub = EventHub()
        self.runner = TaskRunner(config, client, workspace, self.hub)
        self._shutdown = threading.Event()
        self._server: ASCSHTTPServer | None = None

    # -- HTTP server lifecycle ---------------------------------------------

    def bind(self) -> str:
        """Bind the server (non-blocking); returns the URL."""
        self._server = ASCSHTTPServer((self.config.ui_host, self.config.ui_port), self)
        host, port = self._server.server_address[:2]
        return f"http://127.0.0.1:{port}"

    def serve_forever(self) -> None:
        if self._server is not None:
            try:
                self._server.serve_forever(poll_interval=0.5)
            except KeyboardInterrupt:  # pragma: no cover - interactive Ctrl+C
                self.stop()

    def start(self) -> str:
        """Bind and serve in a background thread; returns the URL."""
        url = self.bind()
        thread = threading.Thread(target=self.serve_forever, daemon=True, name="ascs-web")
        thread.start()
        return url

    def stop(self) -> None:
        if self.runner.busy:
            self.runner.cancel()
        self._shutdown.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # -- API payloads -------------------------------------------------------

    def state(self) -> dict[str, Any]:
        snap = self.runner.tracker.snapshot
        data = snap.to_dict()
        result = self.runner.result
        if result is not None:
            data["result"] = {
                "status": result.status,
                "state": result.state,
                "summary": result.summary[:2000],
                "error": result.error[:2000],
                "iterations": result.iterations,
            }
            if result.plan is not None:
                data["result"]["plan"] = result.plan.to_dict()
        data["busy"] = self.runner.busy
        data["mode"] = snap.mode or self.config.mode
        data["version"] = __version__
        data["state_label"] = STATE_LABELS.get(snap.state, snap.state.upper())
        return data

    def status(self) -> dict[str, Any]:
        try:
            self.client.check_connectivity(timeout=3)
            ollama_up = True
        except Exception:
            ollama_up = False
        models = []
        try:
            models = self.client.list_models(timeout=3)
        except Exception:
            pass
        return {
            "server": __version__,
            "uptime_ok": True,
            "ollama": {
                "reachable": ollama_up,
                "base_url": self.config.ollama_base_url,
                "model": self.config.model,
                "installed": models,
                "model_available": self.config.model in models,
            },
            "workspace": str(self.config.workspace),
            "mode": self.config.mode,
            "ui_host": self.config.ui_host,
            "ui_port": self.config.ui_port,
        }


def serve(
    config: AgentConfig,
    client: Any,
    workspace: Workspace,
    *,
    block: bool = True,
) -> str | App:
    """Create the app server and serve (optionally in a background thread).

    Returns the URL when ``block=False``; otherwise runs forever.
    """
    app = App(config, client, workspace)
    url = app.start()
    print(f"A.S.C.S. web UI: {url}   (STOP at any time with Ctrl+C)")
    print(
        f"Mode: {config.mode}   Model: {config.model}   "
        f"Workspace: {config.workspace}"
    )
    if block:
        app.serve_forever()
    return url


__all__ = [
    "App",
    "ASCSHTTPServer",
    "EventHub",
    "TaskRunner",
    "interrupt_thread",
    "serve",
]