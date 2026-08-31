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
import time
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
    """Remove Qwen-style ``<think>...</think>`` blocks."""
    return _THINK_TAG_RE.sub("", content).strip()


def _strip_think_tags_streaming(
    buffer: str,
    new_delta: str,
) -> tuple[str, str]:
    """Strip think blocks while preserving incomplete streaming tags."""
    combined = buffer + new_delta

    if "<think>" not in combined:
        return combined, ""

    if "</think>" in combined:
        cleaned = _THINK_TAG_RE.sub("", combined).strip()

        last_open = cleaned.rfind("<think>")
        if last_open >= 0 and "</think>" not in cleaned[last_open:]:
            return cleaned[:last_open].rstrip(), cleaned[last_open:]

        return cleaned, ""

    last_open = combined.rfind("<think>")
    if last_open >= 0:
        return combined[:last_open].rstrip(), combined[last_open:]

    return combined, ""


def _chat_payload_error(status: int, body: str) -> OllamaError:
    """Convert an Ollama HTTP failure into a typed exception."""
    detail = body.strip().replace("\n", " ")

    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and parsed.get("error"):
            detail = str(parsed["error"])
    except (json.JSONDecodeError, ValueError):
        pass

    if status == 404:
        return OllamaModelNotFoundError(
            status,
            detail or "model not found",
        )

    return OllamaHTTPError(
        status,
        detail or "request failed",
    )


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
        self._active = None

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def _track(self, resp: Any, timeout: int) -> None:
        del timeout

        with self._resp_lock:
            self._active = resp

    def _untrack(self, resp: Any) -> None:
        with self._resp_lock:
            if self._active is resp:
                self._active = None

    def abort_current(self) -> None:
        """Interrupt the currently active HTTP response, if any."""
        with self._resp_lock:
            resp = self._active
            self._active = None

        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout: int,
    ) -> bytes:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Connection": "close",
            },
            method="POST",
        )

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
            except Exception:
                pass

            error = _chat_payload_error(
                exc.code,
                body.decode("utf-8", errors="replace"),
            )

            raise error from exc

        except urllib.error.URLError as exc:
            raise OllamaConnectionError(
                f"cannot reach {url}: {exc.reason}"
            ) from exc

        except TimeoutError as exc:
            raise OllamaTimeoutError(
                f"request to {url} timed out after {timeout}s"
            ) from exc

        except OSError as exc:
            raise OllamaConnectionError(
                f"cannot reach {url}: {exc}"
            ) from exc

    def _get(self, endpoint: str, timeout: int) -> bytes:
        url = f"{self.base_url}{endpoint}"

        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Connection": "close",
            },
            method="GET",
        )

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
            except Exception:
                pass

            detail = body.decode(
                "utf-8",
                errors="replace",
            ).strip()

            raise OllamaHTTPError(
                exc.code,
                detail or f"GET {endpoint} failed",
            ) from exc

        except urllib.error.URLError as exc:
            raise OllamaConnectionError(
                f"cannot reach {url}: {exc.reason}"
            ) from exc

        except TimeoutError as exc:
            raise OllamaTimeoutError(
                f"GET {url} timed out after {timeout}s"
            ) from exc

        except OSError as exc:
            raise OllamaConnectionError(
                f"cannot reach {url}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_chat_body(raw: bytes) -> dict[str, Any]:
        """Parse either normal JSON or Ollama NDJSON."""
        text = raw.decode(
            "utf-8",
            errors="replace",
        ).strip()

        if not text:
            raise OllamaResponseError(
                "Ollama returned an empty response body"
            )

        try:
            obj = json.loads(text)

            if isinstance(obj, dict):
                return obj

        except json.JSONDecodeError:
            pass

        objects: list[dict[str, Any]] = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(obj, dict):
                objects.append(obj)

        if not objects:
            raise OllamaResponseError(
                "Ollama response was not valid JSON: "
                + text[:200].replace("\n", " ")
            )

        merged: dict[str, Any] = {
            "message": {
                "role": "assistant",
                "content": "",
            }
        }

        for obj in objects:
            message = obj.get("message") or {}

            if isinstance(message, dict):
                merged["message"]["content"] += (
                    message.get("content") or ""
                )

            if obj.get("done"):
                merged["done"] = True

                if "done_reason" in obj:
                    merged["done_reason"] = obj["done_reason"]

            for key, value in obj.items():
                if key != "message":
                    merged[key] = value

        return merged

    @staticmethod
    def _parse_version(raw: bytes) -> str:
        try:
            data = json.loads(
                raw.decode(
                    "utf-8",
                    errors="replace",
                )
            )

            if isinstance(data, dict):
                return str(data.get("version") or "")

        except json.JSONDecodeError:
            pass

        return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_connectivity(self, timeout: int = 5) -> bool:
        """Return True when Ollama responds."""
        self._get("/api/version", timeout=timeout)
        return True

    def get_version(self, timeout: int = 5) -> str:
        """Return the Ollama server version."""
        raw = self._get(
            "/api/version",
            timeout=timeout,
        )

        return self._parse_version(raw)

    def list_models(self, timeout: int = 10) -> list[str]:
        """Return installed Ollama model names."""
        raw = self._get(
            "/api/tags",
            timeout=timeout,
        )

        try:
            data = json.loads(
                raw.decode(
                    "utf-8",
                    errors="replace",
                )
            )

        except json.JSONDecodeError as exc:
            raise OllamaResponseError(
                "could not parse /api/tags response"
            ) from exc

        if not isinstance(data, dict):
            raise OllamaResponseError(
                "/api/tags returned an unexpected response"
            )

        models = data.get("models") or []

        if not isinstance(models, list):
            raise OllamaResponseError(
                "/api/tags returned an invalid models field"
            )

        result: list[str] = []

        for model in models:
            if not isinstance(model, dict):
                continue

            name = model.get("name")

            if name:
                result.append(str(name))

        return result

    def is_model_available(
        self,
        model: str | None = None,
        timeout: int = 10,
    ) -> bool:
        """Return whether a model is installed."""
        target = model or self.model

        return target in self.list_models(
            timeout=timeout,
        )

    def warm(self, timeout: int = 120) -> None:
        """Load the configured model with a tiny request."""
        try:
            self.chat(
                [
                    {
                        "role": "user",
                        "content": "Reply with the single word: ok.",
                    }
                ],
                format=None,
                timeout=timeout,
            )

        except (
            OllamaHTTPError,
            OllamaResponseError,
            OllamaTimeoutError,
        ):
            pass

    def ensure_ready(
        self,
        *,
        check_timeout: int = 5,
        warm_timeout: int = 120,
        prewarm: bool = True,
    ) -> dict[str, Any]:
        """Check server availability, model availability, and optionally warm."""
        self.check_connectivity(
            timeout=check_timeout,
        )

        version = self.get_version(
            timeout=check_timeout,
        )

        installed = self.list_models(
            timeout=check_timeout,
        )

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
            self.warm(
                timeout=warm_timeout,
            )
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
        """Send a non-streaming chat request."""
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

        raw = self._post(
            "/api/chat",
            payload,
            timeout or self.request_timeout,
        )

        parsed = self._parse_chat_body(raw)

        if "error" in parsed and not parsed.get("message"):
            raise OllamaResponseError(
                str(parsed["error"])
            )

        message = parsed.get("message") or {}

        if not isinstance(message, dict):
            raise OllamaResponseError(
                "Ollama response contained an invalid message"
            )

        content = message.get("content") or ""

        if not isinstance(content, str):
            content = str(content)

        content = _strip_think_tags(content)

        if not content:
            raise OllamaResponseError(
                "Ollama returned an empty assistant response"
            )

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
        keep_alive: str | None = None,
    ) -> str:
        """Chat with bounded retries for transient local-server failures."""
        attempt = 0
        last_exc: OllamaError | None = None

        while True:
            try:
                return self.chat(
                    messages,
                    format=format,
                    options=options,
                    timeout=timeout,
                    keep_alive=keep_alive,
                )

            except (
                OllamaConnectionError,
                OllamaTimeoutError,
            ) as exc:
                last_exc = exc

            except OllamaHTTPError as exc:
                if exc.status < 500:
                    raise

                last_exc = exc

            if last_exc is None:
                raise OllamaError(
                    "Ollama request failed without an exception"
                )

            if attempt >= max_retries:
                raise last_exc

            if should_stop is not None and should_stop():
                raise last_exc

            attempt += 1
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
        """Yield visible assistant text as Ollama generates it."""
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
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/x-ndjson, application/json",
                "Connection": "close",
            },
            method="POST",
        )

        request_timeout = timeout or self.request_timeout

        try:
            with urllib.request.urlopen(
                req,
                timeout=request_timeout,
            ) as resp:
                self._track(
                    resp,
                    request_timeout,
                )

                think_buffer = ""

                try:
                    for raw_line in resp:
                        line = raw_line.decode(
                            "utf-8",
                            errors="replace",
                        ).strip()

                        if not line:
                            continue

                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if not isinstance(obj, dict):
                            continue

                        if obj.get("error"):
                            raise OllamaResponseError(
                                str(obj["error"])
                            )

                        message = obj.get("message") or {}

                        if not isinstance(message, dict):
                            continue

                        delta = message.get("content") or ""

                        if not isinstance(delta, str):
                            delta = str(delta)

                        visible, think_buffer = (
                            _strip_think_tags_streaming(
                                think_buffer,
                                delta,
                            )
                        )

                        if visible:
                            yield visible

                finally:
                    self._untrack(resp)

        except urllib.error.HTTPError as exc:
            body = b""

            try:
                body = exc.read()
            except Exception:
                pass

            raise _chat_payload_error(
                exc.code,
                body.decode(
                    "utf-8",
                    errors="replace",
                ),
            ) from exc

        except urllib.error.URLError as exc:
            raise OllamaConnectionError(
                f"cannot reach {url}: {exc.reason}"
            ) from exc

        except TimeoutError as exc:
            raise OllamaTimeoutError(
                f"request to {url} timed out after {request_timeout}s"
            ) from exc

        except OSError as exc:
            raise OllamaConnectionError(
                f"cannot reach {url}: {exc}"
            ) from exc
