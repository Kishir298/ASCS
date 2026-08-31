"""Isolated Ollama HTTP API client.

All provider-specific HTTP logic lives here. The rest of the agent talks to
``OllamaClient`` and never issues HTTP requests itself.

Implementation notes:
    * Uses only the standard library (``urllib``), so the agent has zero
      runtime dependencies.
    * Requests are sent with ``"stream": false`` explicitly. Some Ollama
      builds stream NDJSON-lines even when the field is omitted, so responses
      are parsed robustly whether they arrive as one JSON object or as chunks.
"""

from __future__ import annotations

import json
import re as _re
import threading
import urllib.error
import urllib.request
from typing import Any, Iterator


class OllamaError(Exception):
    """Base class for all Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Could not reach the Ollama server at all."""


class OllamaTimeoutError(OllamaError):
    """The request exceeded its timeout."""


class OllamaHTTPError(OllamaError):
    """The server returned a non-2xx HTTP status."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"Ollama HTTP {status}: {detail}")


class OllamaModelNotFoundError(OllamaHTTPError):
    """The requested model is not installed on the server."""


class OllamaResponseError(OllamaError):
    """The server returned something we could not interpret."""


_THINK_TAG_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)


def _strip_think_tags(content: str) -> str:
    """Remove ``<think>...</think>`` blocks that Qwen3 and similar models emit.

    The reasoning tokens inside ``<think>`` are useful to the model internally
    but waste the context budget and can break downstream JSON parsing.
    Stripping them keeps the visible response clean.
    """
    return _THINK_TAG_RE.sub("", content).strip()


def _strip_think_tags_streaming(buffer: str, new_delta: str) -> tuple[str, str]:
    """Strip ``<think>`` tags across streaming deltas.

    Returns ``(visible_text, updated_buffer)`` where ``visible_text`` is
    content safe to yield and ``updated_buffer`` carries any unfinished
    think-tag state.

    The web UI calls this for each streaming delta so partial think tags
    that span multiple chunks are handled correctly.
    """
    combined = buffer + new_delta
    # Fast path: no think-tag content at all.
    if "<think>" not in combined:
        return combined, ""
    # Slow path: check for a completed think block.
    if "</think>" in combined:
        cleaned = _THINK_TAG_RE.sub("", combined).strip()
        # After stripping, check if an incomplete opening tag remains
        # (e.g. a new think block started but hasn't closed yet).
        last_open = cleaned.rfind("<think>")
        if last_open >= 0 and "</think>" not in cleaned[last_open:]:
            # Keep only the part before the unfinished tag.
            return cleaned[:last_open].rstrip(), cleaned[last_open:]
        return cleaned, ""
    # No closing tag yet: buffer everything (might be inside a think block).
    last_open = combined.rfind("<think>")
    if last_open >= 0:
        return combined[:last_open].rstrip(), combined[last_open:]
    # No opening tag — this shouldn't happen if <think> was in combined,
    # but be safe.
    return combined, ""


def _chat_payload_error(status: int, body: str) -> OllamaError:
    """Turn /api/chat failure metadata into a typed exception."""
    detail = body.strip().replace("\n", " ")
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and parsed.get("error"):
            detail = str(parsed["error"])
    except (json.JSONDecodeError, ValueError):
        pass
    if status == 404:
        return OllamaModelNotFoundError(status, detail or "model not found")
    return OllamaHTTPError(status, detail or "request failed")


class OllamaClient:
    """Minimal HTTP client for a local Ollama server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:14b",
        request_timeout: int = 600,
        keep_alive: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.request_timeout = request_timeout
        self.keep_alive = keep_alive
        self._resp_lock = threading.Lock()
        self._active = None  # active HTTPResponse, for out-of-band cancellation

    # -- cancellation support ----------------------------------------------

    def _track(self, resp, timeout: int) -> None:
        with self._resp_lock:
            self._active = resp

    def _untrack(self, resp) -> None:
        with self._resp_lock:
            if self._active is resp:
                self._active = None

    def abort_current(self) -> None:
        """Close any in-flight response so a blocked read is interrupted.

        Used by the UI/TaskRunner to cancel a running Ollama generation: the
        closing socket makes the worker thread's blocking read raise (OSError)
        instead of waiting for the model to finish.
        """
        with self._resp_lock:
            resp = self._active
            self._active = None
        if resp is not None:
            try:
                resp.close()
            except Exception:  # pragma: no cover - defensive
                pass

    # -- low level ---------------------------------------------------------

    def _post(self, endpoint: str, payload: dict[str, Any], timeout: int) -> bytes:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Explicitly close the connection after each request. On Windows this
        # avoids a keep-alive RST race with local servers that otherwise aborts
        # reads with ConnectionAbortedError; it also stays robust across tools.
        req.add_header("Connection", "close")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                self._track(resp, timeout)
                try:
                    return resp.read()
                finally:
                    self._untrack(resp)
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read()
            except Exception:  # pragma: no cover - defensive
                pass
            error = _chat_payload_error(
                exc.code, body.decode("utf-8", errors="replace")
            )
            raise error from exc
        except urllib.error.URLError as exc:
            raise OllamaConnectionError(f"cannot reach {url}: {exc.reason}") from exc
        except TimeoutError as exc:  # py3.10+ raises TimeoutError subclass
            raise OllamaTimeoutError(
                f"request to {url} timed out after {timeout}s"
            ) from exc
        except OSError as exc:
            raise OllamaConnectionError(f"cannot reach {url}: {exc}") from exc

    def _get(self, endpoint: str, timeout: int) -> bytes:
        url = f"{self.base_url}{endpoint}"
        req = urllib.request.Request(url)
        req.add_header("Connection", "close")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                self._track(resp, timeout)
                try:
                    return resp.read()
                finally:
                    self._untrack(resp)
        except urllib.error.HTTPError as exc:
            raise OllamaHTTPError(exc.code, f"GET {endpoint} failed") from exc
        except urllib.error.URLError as exc:
            raise OllamaConnectionError(f"cannot reach {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise OllamaTimeoutError(f"GET {url} timed out after {timeout}s") from exc
        except OSError as exc:
            raise OllamaConnectionError(f"cannot reach {url}: {exc}") from exc

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _parse_chat_body(raw: bytes) -> dict[str, Any]:
        """Parse /api/chat output (single JSON object or NDJSON lines)."""
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            raise OllamaResponseError("Ollama returned an empty response body")
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        objects = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not objects:
            raise OllamaResponseError(
                "Ollama response was not valid JSON: "
                + text[:200].replace("\n", " ")
            )
        merged = {"message": {"role": "assistant", "content": ""}}
        for obj in objects:
            msg = obj.get("message") or {}
            merged["message"]["content"] += msg.get("content") or ""
            if obj.get("done"):
                merged["done"] = True
                merged.setdefault("done_reason", obj.get("done_reason"))
            merged.update({k: v for k, v in obj.items() if k not in ("message",)})
        return merged

    # -- public API --------------------------------------------------------

    def check_connectivity(self, timeout: int = 5) -> bool:
        """Return True if the Ollama server answers. Raises on failure."""
        self._get("/api/version", timeout=timeout)
        return True

    def list_models(self, timeout: int = 10) -> list[str]:
        raw = self._get("/api/tags", timeout=timeout)
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise OllamaResponseError("could not parse /api/tags response") from exc
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    def is_model_available(self, model: str | None = None, timeout: int = 10) -> bool:
        target = model or self.model
        return target in self.list_models(timeout=timeout)

    def warm(self, timeout: int = 120) -> None:
        """Probe the model with a tiny request to trigger (cold) model load.

        Local models such as ``qwen3:14b`` can take tens of seconds to
        load on first use. Warming up eagerly makes the first real step reach
        the model instead of paying the load latency mid-task. A slow warm is
        *not* an error; ``OllamaError`` subclasses still propagate.
        """
        try:
            self.chat(
                [{"role": "user", "content": "Reply with the single word: ok."}],
                format=None,
                timeout=timeout,
            )
        except (OllamaHTTPError, OllamaResponseError, OllamaTimeoutError):
            # The server answered but choked on our tiny ping; that still means
            # the model finished loading. Treat it as warm enough.
            pass

    def ensure_ready(self, *, check_timeout: int = 5, warm_timeout: int = 120, prewarm: bool = True) -> dict[str, Any]:
        """Run connectivity + model availability checks, optionally prewarming.

        Returns a small report dict:
            {"reachable": bool, "version": str, "model": str,
             "available": bool, "installed": list[str], "warmed": bool}

        Raises ``OllamaConnectionError`` / ``OllamaHTTPError`` when the server
        cannot be reached at all.
        """
        self.check_connectivity(timeout=check_timeout)
        raw = self._get("/api/version", timeout=check_timeout)
        try:
            version = (json.loads(raw.decode("utf-8", errors="replace")) or {}).get("version", "")
        except json.JSONDecodeError:
            version = ""
        installed = self.list_models(timeout=check_timeout)
        available = self.model in installed
        report: dict[str, Any] = {
            "reachable": True,
            "version": version,
            "model": self.model,
            "available": available,
            "installed": installed,
            "warmed": False,
        }
        if prewarm and available:
            self.warm(timeout=warm_timeout)
            report["warmed"] = True
        return report

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        format: str | None = "json",
        options: dict[str, Any] | None = None,
        timeout: int | None = None,
        keep_alive: str | None = None,
    ) -> str:
        """Send a chat request and return the assistant's text content.

        ``keep_alive`` keeps the loaded model resident between calls so
        separate agent steps don't pay a cold-reload penalty (e.g. ``"30m"``).
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options
        if keep_alive is None:
            keep_alive = self.keep_alive
        if keep_alive:
            payload["keep_alive"] = keep_alive
        raw = self._post("/api/chat", payload, timeout or self.request_timeout)
        parsed = self._parse_chat_body(raw)
        message = parsed.get("message") or {}
        content = message.get("content") or ""
        content = _strip_think_tags(content)
        if not content.strip() and "error" in parsed:
            raise OllamaResponseError(str(parsed["error"]))
        if not content.strip():
            raise OllamaResponseError("Ollama returned an empty assistant response")
        return content

    def chat_resilient(
        self,
        messages: list[dict[str, str]],
        *,
        format: str | None = "json",
        options: dict[str, Any] | None = None,
        timeout: int | None = None,
        max_retries: int = 2,
        backoff_s: float = 2.0,
        should_stop: Any = None,
    ) -> str:
        """Call :meth:`chat`, retrying transient failures with backoff.

        A local Ollama server can drop a connection while (re)loading a large
        model (e.g. ``qwen3:14b``) or return a transient 5xx under load. These
        are *retryable*; a missing model or an unparseable response is not.
        Retries are bounded and respect ``should_stop`` so the operator can
        cancel.

        Returns the assistant content on success; raises the last error when
        retries are exhausted or the failure is non-transient.
        """
        import time

        attempt = 0
        last_exc: OllamaError | None = None
        while True:
            try:
                return self.chat(
                    messages, format=format, options=options, timeout=timeout
                )
            except (OllamaConnectionError, OllamaTimeoutError) as exc:
                last_exc = exc
            except OllamaHTTPError as exc:
                # Retry only transient server errors (5xx), never 4xx.
                if not exc.status or exc.status < 500:
                    raise
                last_exc = exc
            if attempt >= max_retries:
                raise last_exc
            attempt += 1
            if should_stop is not None and should_stop():
                raise last_exc
            time.sleep(backoff_s * attempt)

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        format: str | None = "json",
        options: dict[str, Any] | None = None,
        timeout: int | None = None,
        keep_alive: str | None = None,
    ) -> Iterator[str]:
        """Yield assistant content deltas as they arrive.

        The caller is responsible for wrapping the iteration to handle
        ``KeyboardInterrupt`` gracefully.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options
        if keep_alive is None:
            keep_alive = self.keep_alive
        if keep_alive:
            payload["keep_alive"] = keep_alive
        url = f"{self.base_url}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="POST",
        )
        req.add_header("Connection", "close")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.request_timeout) as resp:
                self._track(resp, timeout or self.request_timeout)
                think_buffer = ""
                try:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        delta = (obj.get("message") or {}).get("content") or ""
                        visible, think_buffer = _strip_think_tags_streaming(
                            think_buffer, delta,
                        )
                        if visible:
                            yield visible
                finally:
                    self._untrack(resp)
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read()
            except Exception:  # pragma: no cover
                pass
            raise _chat_payload_error(
                exc.code, body.decode("utf-8", errors="replace")
            ) from exc
        except urllib.error.URLError as exc:
            raise OllamaConnectionError(
                f"cannot reach {url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise OllamaTimeoutError(
                f"request to {url} timed out after {timeout or self.request_timeout}s"
            ) from exc
        except OSError as exc:
            raise OllamaConnectionError(f"cannot reach {url}: {exc}") from exc