"""Tests for tool-call validation and the "never crash" guarantee.

The agent must turn malformed tool calls into concise failures the model can
recover from, rather than letting them crash the loop.
"""

from __future__ import annotations

import pytest

from agent.tools import ToolValidationError, execute_tool, validate_tool_call
from agent.workspace import Workspace


def test_unknown_tool_validation_raises():
    with pytest.raises(ToolValidationError):
        validate_tool_call("nope", {})


def test_non_string_tool_name():
    with pytest.raises(ToolValidationError):
        validate_tool_call(123, {})


def test_arguments_must_be_object():
    with pytest.raises(ToolValidationError):
        validate_tool_call("read_file", "not-a-dict")


def test_missing_required_arg():
    with pytest.raises(ToolValidationError):
        validate_tool_call("write_file", {"path": "x.py"})  # missing content


def test_extra_arguments_are_warned_not_fatal(tmp_path, config):
    cfg = config
    # validate_tool_call returns _warnings for unknown args
    checked = validate_tool_call("git_status", {"bogus": 1})
    assert "_warnings" in checked


def test_execute_tool_never_raises_on_logic_errors(tmp_path, config):
    # Even deeply invalid calls produce a ToolResult, not an exception.
    bad_calls = [
        ("nope", {}),
        ("read_file", {}),
        ("read_file", {"path": 123}),
        ("run_command", {"command": ""}),
        ("write_file", {"path": "../escape.py", "content": "x"}),
    ]
    for name, args in bad_calls:
        result = execute_tool(name, args, Workspace(tmp_path), config)
        assert result is not None
        assert isinstance(result, ToolResult)
        assert not result.ok


def test_unknown_tool_result_message():
    result = execute_tool("frobnicate", {}, None, None)
    assert not result.ok
    assert "Valid tools" in result.output


def test_valid_call_returns_ok_result(tmp_path, config):
    result = execute_tool(
        "write_file", {"path": "ok.py", "content": "x=1"}, Workspace(tmp_path), config
    )
    assert result.ok
