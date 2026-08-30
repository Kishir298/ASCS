"""Data models and lenient JSON parsing for model responses.

The agent uses a strict JSON response contract:

    * Tool use:    {"comment": "...", "tool": "<name>", "arguments": { ... }}
    * Completion:  {"done": true, "summary": "..."}

Parsing is tolerant of markdown fences, leading/trailing prose, and full
objects embedded in larger text. Everything else is surfaced as a concise
error so the loop can hand it back to the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


def truncate(text: str, max_chars: int, marker: str = "... [truncated]") -> str:
    """Truncate ``text`` to ``max_chars``, appending ``marker`` when clipped."""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(marker):
        return marker[:max_chars]
    head = max_chars - len(marker)
    return text[:head].rstrip() + marker


def _strip_fences(text: str) -> str:
    text = text.strip()
    lines = text.splitlines()
    if not lines:
        return ""
    if re.match(r"^```(?:json)?\s*$", lines[0], re.IGNORECASE):
        if lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return text


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Find the first top-level JSON object in ``text``."""
    text = _strip_fences(text)
    start = text.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


@dataclass(frozen=True)
class ModelReply:
    """A parsed response from the model."""

    comment: str = ""
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    summary: str = ""
    error: str | None = None
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_validation_error(self) -> str:
        return f"Invalid response: {self.error}"


def parse_model_reply(text: str) -> ModelReply:
    """Parse raw model text into a validated ``ModelReply``.

    Never raises; any failure is reported via the ``error`` field.
    """
    raw = text
    obj = _extract_json_object(text)
    if obj is None:
        return ModelReply(
            error="Expected a JSON object like "
            '{"tool": "<name>", "arguments": {...}} or {"done": true, "summary": "..."}. '
            f"Could not parse a JSON object from the response.",
            raw=raw,
        )

    comment = obj.get("comment")
    if not isinstance(comment, str):
        comment = str(comment) if comment is not None else ""
    summary = obj.get("summary")
    if not isinstance(summary, str):
        summary = str(summary) if summary is not None else ""

    done = obj.get("done") in (True, "true")
    tool = obj.get("tool")
    if tool is not None and not isinstance(tool, str):
        tool = str(tool)
    arguments = obj.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return ModelReply(
            comment=comment,
            error='"arguments" must be a JSON object.',
            raw=raw,
        )
    # Ignore/filter unknown keys to keep the contract strict but harmless.
    if not isinstance(tool, str) or not tool.strip():
        tool = None

    if done and not tool:
        if not summary:
            summary = comment or "Task completed."
        return ModelReply(comment=comment, done=True, summary=summary, raw=raw)

    if not done and not tool:
        return ModelReply(
            comment=comment,
            error="Response must contain either a valid "
            '"tool" (with "arguments") or "done": true.',
            raw=raw,
        )

    if not done:
        return ModelReply(
            comment=comment, tool=tool, arguments=arguments, raw=raw
        )

    # done AND tool: prefer the tool; keep summary for the eventual report.
    return ModelReply(comment=comment, tool=tool, arguments=arguments, raw=raw)


class Plan:
    """A structured implementation plan produced by the ``set_plan`` tool.

    Tolerates a range of model outputs (list of strings, list of ``{"step":
    ..., "detail": ...}`` dicts, or a plain multi-line string) and renders
    them deterministically for logs, the UI, and BUILD-mode review.
    """

    def __init__(self, steps: list[str], goal: str = "") -> None:
        self.steps = [str(s) for s in steps]
        self.goal = goal

    @classmethod
    def from_value(cls, value: Any) -> "Plan":
        """Build a Plan from a relaxed JSON value; never raises."""
        goal = ""
        raw_steps: list[str] = []
        if isinstance(value, str):
            raw_steps = [ln.strip() for ln in value.splitlines() if ln.strip()]
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    raw_steps.append(item.strip())
                elif isinstance(item, dict):
                    step = item.get("step")
                    if not step and "detail" in item:
                        step = item["detail"]
                    elif isinstance(step, str) and item.get("detail"):
                        raw_steps.append(f"{step}: {item['detail']}")
                    if step:
                        raw_steps.append(str(step).strip())
        elif isinstance(value, dict):
            goal = value.get("goal") or ""
            plan = value.get("plan")
            return cls.from_value(plan)
        if not raw_steps:
            return cls(["No explicit plan provided."], goal)
        return cls(raw_steps, goal)

    @property
    def ok(self) -> bool:
        return bool(self.steps)

    def to_text(self) -> str:
        lines = [f"Goal: {self.goal}"] if self.goal else []
        for i, step in enumerate(self.steps, 1):
            lines.append(f"{i}. {step}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "steps": self.steps}


def tool_result_message(tool_result: "ToolResult") -> dict[str, str]:
    """Format a ToolResult as a chat message for the model.

    Tool results are sent with the ``user`` role for maximum compatibility
    with small local models (some builds mishandle the ``tool`` role).
    """
    status = "OK" if tool_result.ok else "FAILED"
    content = (
        f"Tool result for {tool_result.name} ({status}):\n{tool_result.output}"
    )
    return {"role": "user", "content": content}


class ToolResult:
    """Outcome of executing a tool call."""

    def __init__(
        self,
        name: str,
        output: str,
        ok: bool = True,
        note: str = "",
    ) -> None:
        self.name = name
        self.output = output
        self.ok = ok
        self.note = note

    @property
    def error(self) -> bool:
        return not self.ok

    def to_model_text(self) -> str:
        if self.ok:
            if not self.note:
                return f"Tool {self.name} succeeded.\n{self.output}"
            return (
                f"Tool {self.name} succeeded ({self.note}).\n{self.output}"
            )
        return (
            f"Tool {self.name} FAILED: {self.output}"
        )


__all__ = [
    "Plan",
    "ModelReply",
    "ToolResult",
    "parse_model_reply",
    "tool_result_message",
    "truncate",
]