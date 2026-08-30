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
        model: str = "qwen2.5-coder:7b",
        request_timeout: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.request_timeout = request_timeout

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
                return resp.read()
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
                return resp.read()
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

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        format: str | None = "json",
        options: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> str:
        """Send a chat request and return the assistant's text content."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options
        raw = self._post("/api/chat", payload, timeout or self.request_timeout)
        parsed = self._parse_chat_body(raw)
        message = parsed.get("message") or {}
        content = message.get("content") or ""
        if not content.strip() and "error" in parsed:
            raise OllamaResponseError(str(parsed["error"]))
        if not content.strip():
            raise OllamaResponseError("Ollama returned an empty assistant response")
        return content

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        format: str | None = "json",
        options: dict[str, Any] | None = None,
        timeout: int | None = None,
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
        url = f"{self.base_url}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="POST",
        )
        req.add_header("Connection", "close")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.request_timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = (obj.get("message") or {}).get("content") or ""
                    if delta:
                        yield delta
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