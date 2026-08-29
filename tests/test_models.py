"""Tests for model-reply parsing and ToolResult formatting."""

from __future__ import annotations

from agent.models import ToolResult, parse_model_reply, tool_result_message, truncate


def test_parse_tool_call():
    r = parse_model_reply(
        '{"comment": "listing", "tool": "list_directory", "arguments": {"path": "."}}'
    )
    assert r.ok
    assert r.tool == "list_directory"
    assert r.arguments == {"path": "."}
    assert not r.done


def test_parse_done():
    r = parse_model_reply('{"done": true, "summary": "all done"}')
    assert r.ok
    assert r.done
    assert r.summary == "all done"


def test_parse_done_without_summary():
    r = parse_model_reply('{"done": true}')
    assert r.done
    assert r.summary  # defaults to comment/task-completed


def test_parse_done_stringified():
    r = parse_model_reply('{"done": "true", "summary": "s"}')
    assert r.done


def test_parse_fenced_json():
    r = parse_model_reply(
        '```json\n{"tool": "read_file", "arguments": {"path": "a.py"}}\n```'
    )
    assert r.ok
    assert r.tool == "read_file"


def test_parse_json_embedded_in_prose():
    r = parse_model_reply(
        "Sure, here is my plan.\n{\"tool\": \"git_status\", \"arguments\": {}}\nHope that helps."
    )
    assert r.ok
    assert r.tool == "git_status"


def test_parse_no_json():
    r = parse_model_reply("I am thinking about the task.")
    assert not r.ok
    assert r.error is not None


def test_parse_arguments_not_object():
    r = parse_model_reply('{"tool": "read_file", "arguments": "nope"}')
    assert not r.ok
    assert "arguments" in r.error


def test_parse_neither_tool_nor_done():
    r = parse_model_reply('{"comment": "hi"}')
    assert not r.ok
    assert "tool" in r.error


def test_parse_unknown_keys_ignored():
    r = parse_model_reply(
        '{"tool": "read_file", "arguments": {"path": "a"}, "bogus": 123}'
    )
    assert r.ok
    assert r.tool == "read_file"


def test_truncate_within_limit():
    assert truncate("hello", 100) == "hello"


def test_truncate_over_limit():
    out = truncate("a" * 100, 20)
    assert len(out) == 20
    assert "truncated" in out


def test_tool_result_message_ok():
    tr = ToolResult("read_file", "some output")
    msg = tool_result_message(tr)
    assert msg["role"] == "user"
    assert "read_file" in msg["content"]
    assert "some output" in msg["content"]


def test_tool_result_message_failed():
    tr = ToolResult("apply_patch", "boom", ok=False)
    msg = tool_result_message(tr)
    assert "FAILED" in msg["content"]


def test_tool_result_to_model_text():
    tr = ToolResult("write_file", "wrote", ok=True)
    assert "succeeded" in tr.to_model_text()
    tr2 = ToolResult("run_command", "err", ok=False)
    assert "FAILED" in tr2.to_model_text()
