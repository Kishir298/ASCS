"""The autonomous agent loop.

Drives: user task -> model analysis -> tool selection -> execution -> result
-> model analysis -> ... until completion, iteration limit, interruption, or
a fatal error. In AUTO mode every valid tool call executes without prompting;
in SAFE mode modifications and commands require operator approval.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import AgentConfig
from .models import ToolResult, parse_model_reply, tool_result_message
from .ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from .prompts import (
    malformed_feedback,
    system_prompt,
    task_message,
    tool_error_feedback,
)
from .tools import execute_tool, get_tool_spec
from .workspace import Workspace

DEFAULT_APPROVER: Callable[[str], bool] = lambda desc: (
    input(f"[SAFE] Approve: {desc} [y/N] ").strip().lower() in ("y", "yes")
)

MODIFY_TOOLS = {"write_file", "apply_patch", "run_command"}

RETRY_PROMPT = (
    "The model returned an unusable response. Reply with ONLY a valid JSON "
    "tool call or done object as instructed."
)


@dataclass
class LoopResult:
    status: str = "interrupted"  # completed|max_iterations|interrupted|fatal|malformed
    summary: str = ""
    iterations: int = 0
    steps: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def is_complete(self) -> bool:
        return self.status == "completed"


class AgentLoop:
    """Runs one agent session against a workspace."""

    def __init__(
        self,
        config: AgentConfig,
        client: OllamaClient,
        workspace: Workspace,
        *,
        approver: Callable[[str], bool] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.ws = workspace
        self.log = log or (lambda _msg: None)
        self.approver = approver if approver is not None else DEFAULT_APPROVER
        self._steps: list[str] = []
        self._messages: list[dict[str, str]] = []
        self._malformed_count = 0
        self._last_call: tuple[Any, ...] | None = None
        self._last_ok = True
        self._repeat_count = 0

    # -- logging ------------------------------------------------------------

    def _step(self, text: str) -> None:
        self._steps.append(text)
        self.log(text)

    # -- main entry ---------------------------------------------------------

    def run(self, task: str) -> LoopResult:
        self._messages = [system_prompt(self.config), task_message(task)]
        self._step(f"Task received: {task[:300]}")
        self._step(f"Workspace: {self.ws.root}")
        self._step(
            f"Model: {self.client.model}  Mode: {self.config.mode}  "
            f"Max iterations: {self.config.max_iterations}"
        )

        iteration = 0
        try:
            while iteration < self.config.max_iterations:
                iteration += 1
                self._step(f"[{iteration:02d}] Asking model for the next step...")

                try:
                    reply_text = self.client.chat(
                        self._messages_for_request(), format="json"
                    )
                except OllamaResponseError as exc:
                    self._step(f"[{iteration:02d}] Unusable model response: {exc}")
                    self._messages.append(
                        {"role": "assistant", "content": f"(empty/invalid response: {exc})"}
                    )
                    self._messages.append({"role": "user", "content": RETRY_PROMPT})
                    if self._bump_malformed(iteration):
                        return self._finish(
                            "malformed",
                            "Model repeatedly produced unusable responses.",
                            iteration,
                            error=str(exc),
                        )
                    continue

                reply = parse_model_reply(reply_text)

                if reply.error is not None:
                    self._step(f"[{iteration:02d}] Model reply rejected: {reply.error}")
                    self._messages.append({"role": "assistant", "content": reply_text})
                    self._messages.append(malformed_feedback(reply.error))
                    if self._bump_malformed(iteration):
                        return self._finish(
                            "malformed",
                            "Model repeatedly failed to produce a valid response.",
                            iteration,
                            error=reply.error,
                        )
                    continue

                if reply.comment:
                    self._step(f"[{iteration:02d}] {reply.comment[:300]}")

                if reply.done:
                    self._step("[done] Model reports the task is complete.")
                    return self._finish("completed", reply.summary, iteration)

                if reply.tool not in self.config.tools:
                    valid = ", ".join(self.config.tools)
                    self._step(
                        f"[{iteration:02d}] Tool '{reply.tool}' is not enabled; "
                        f"enabled tools: {valid}"
                    )
                    self._messages.append({"role": "assistant", "content": reply_text})
                    self._messages.append(
                        tool_error_feedback(
                            reply.tool or "",
                            f"tool not enabled in this session (enabled: {valid}).",
                        )
                    )
                    continue

                if self._is_repeated_call(reply.tool, reply.arguments):
                    # Only identical, CONSECUTIVE, FAILING calls are treated as
                    # a stuck loop. Identical successful calls are harmless.
                    if self._last_ok:
                        self._repeat_count = 0
                    else:
                        self._repeat_count += 1
                        if self._repeat_count >= 2:
                            self._step(
                                f"[{iteration:02d}] Identical failing call repeated too many times."
                            )
                            return self._finish(
                                "fatal",
                                "The model repeated an identical failing tool call; "
                                "stopping to avoid an infinite loop.",
                                iteration,
                            )
                else:
                    self._repeat_count = 0

                spec = get_tool_spec(reply.tool)
                self._last_call = (reply.tool, tuple(sorted(reply.arguments.items())))
                display_args = _json.dumps(reply.arguments, sort_keys=True)

                if self.config.is_safe_mode and reply.tool in MODIFY_TOOLS:
                    if not self.approver(f"{reply.tool} {display_args}"):
                        result = ToolResult(
                            reply.tool,
                            "Operator declined the action in SAFE mode.",
                            ok=False,
                        )
                        self._step(f"[{iteration:02d}] SAFE mode: action declined by operator.")
                    else:
                        result = execute_tool(reply.tool, reply.arguments, self.ws, self.config)
                        self._step(f"[{iteration:02d}] {self._result_line(reply.tool, result)}")
                else:
                    result = execute_tool(reply.tool, reply.arguments, self.ws, self.config)
                    self._step(f"[{iteration:02d}] {self._result_line(reply.tool, result)}")
                    if self.config.verbose and result.ok:
                        self.log(self._preview_output(result))

                if result.ok:
                    self._malformed_count = 0
                self._last_ok = result.ok

                self._messages.append({"role": "assistant", "content": reply_text})
                self._messages.append(tool_result_message(result))

            return self._finish(
                "max_iterations",
                f"Stopped after {iteration} iterations "
                f"(AGENT_MAX_ITERATIONS={self.config.max_iterations}).",
                iteration,
            )
        except KeyboardInterrupt:
            self._step("[interrupt] Stopped by user (Ctrl+C).")
            return self._finish("interrupted", "Interrupted by the user.", iteration + 1)
        except OllamaTimeoutError as exc:
            return self._finish("fatal", "Ollama request timed out.", iteration, error=str(exc))
        except OllamaConnectionError as exc:
            return self._finish("fatal", "Ollama is unavailable.", iteration, error=str(exc))
        except OllamaModelNotFoundError as exc:
            return self._finish(
                "fatal",
                f"Model '{self.client.model}' is not installed on the Ollama server.",
                iteration,
                error=str(exc),
            )
        except OllamaError as exc:
            return self._finish(
                "fatal", "Ollama request failed.", iteration, error=str(exc)
            )

    # -- internals ----------------------------------------------------------

    def _bump_malformed(self, iteration: int) -> bool:
        """Count a malformed/unusable reply; True when the retry limit is hit."""
        self._malformed_count += 1
        return self._malformed_count >= self.config.malformed_retry_limit

    def _messages_for_request(self) -> list[dict[str, str]]:
        messages = list(self._messages)
        budget = self.config.context_budget_chars
        total = sum(len(m.get("content") or "") for m in messages)
        while total > budget and len(messages) > 2:
            removed = messages.pop(2)
            total -= len(removed.get("content") or "")
        return messages

    def _is_repeated_call(self, tool: str, arguments: dict[str, Any]) -> bool:
        key = (tool, tuple(sorted(arguments.items())))
        if self._last_call == key:
            return True
        return False

    def _result_line(self, tool: str, result: ToolResult) -> str:
        head = result.note or ("succeeded" if result.ok else "FAILED")
        return f"[result] {tool}: {head}"

    def _preview_output(self, result: ToolResult) -> str:
        body = result.output
        if len(body) > 600:
            body = body[:597] + "..."
        return f"[output] {result.name}: {body}"

    def _finish(
        self, status: str, summary: str, iterations: int, error: str = ""
    ) -> LoopResult:
        return LoopResult(
            status=status,
            summary=summary,
            iterations=iterations,
            steps=list(self._steps),
            error=error,
        )


def run_agent(
    config: AgentConfig,
    client: OllamaClient,
    task: str,
    *,
    approver: Callable[[str], bool] | None = None,
    log: Callable[[str], None] | None = None,
) -> LoopResult:
    workspace = Workspace(config.workspace)
    loop = AgentLoop(config, client, workspace, approver=approver, log=log)
    return loop.run(task)