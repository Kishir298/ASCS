"""Tests for the individual tools (read/write/list/search/patch/run/git)."""

from __future__ import annotations

import sys

import pytest

from agent.models import ToolResult
from agent.tools import execute_tool, get_tool_spec, tool_schema_text, validate_tool_call
from agent.workspace import Workspace, WorkspaceError
from agent.tools import ToolValidationError


def test_all_required_tools_registered():
    expected = {
        "list_directory",
        "read_file",
        "search_files",
        "write_file",
        "apply_patch",
        "run_command",
        "delete_file",
        "move_file",
        "copy_file",
        "set_plan",
        "inspect_environment",
        "git_status",
        "git_diff",
    }
    assert expected <= set(get_tool_spec.__globals__["TOOL_SPECS"])


def test_tool_schema_text_mentions_all():
    text = tool_schema_text()
    for name in (
        "list_directory",
        "read_file",
        "search_files",
        "write_file",
        "apply_patch",
        "run_command",
        "delete_file",
        "move_file",
        "copy_file",
        "set_plan",
        "inspect_environment",
        "git_status",
        "git_diff",
    ):
        assert name in text


def test_write_then_read(tmp_path, config):
    r = execute_tool("write_file", {"path": "a/b.py", "content": "print('hi')\n"}, Workspace(tmp_path), config)
    assert r.ok
    f = tmp_path / "a" / "b.py"
    assert f.read_text() == "print('hi')\n"
    r2 = execute_tool("read_file", {"path": "a/b.py"}, Workspace(tmp_path), config)
    assert r2.ok
    assert "print('hi')" in r2.output


def test_read_missing_file(tmp_path, config):
    r = execute_tool("read_file", {"path": "nope.txt"}, Workspace(tmp_path), config)
    assert not r.ok


def test_read_binary_suffix_rejected(tmp_path, config):
    (tmp_path / "bin.exe").write_bytes(b"\x00\x01\x02")
    r = execute_tool("read_file", {"path": "bin.exe"}, Workspace(tmp_path), config)
    assert not r.ok
    assert "binary" in r.output.lower()


def test_read_binary_content_rejected(tmp_path, config):
    (tmp_path / "data.txt").write_bytes(b"abc\x00def")
    r = execute_tool("read_file", {"path": "data.txt"}, Workspace(tmp_path), config)
    assert not r.ok


def test_list_directory(tmp_path, config):
    (tmp_path / "main.py").write_text("x")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("y")
    r = execute_tool("list_directory", {}, Workspace(tmp_path), config)
    assert r.ok
    assert "main.py" in r.output
    assert "docs" in r.output


def test_list_directory_recursive(tmp_path, config):
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    (tmp_path / "sub" / "deep" / "f.py").write_text("x")
    r = execute_tool(
        "list_directory", {"recursive": True}, Workspace(tmp_path), config
    )
    assert r.ok
    assert "deep" in r.output
    assert "f.py" in r.output


def test_search_files(tmp_path, config):
    (tmp_path / "a.py").write_text("def main():\n    pass\n")
    (tmp_path / "b.txt").write_text("def main():\n    pass\n")
    r = execute_tool(
        "search_files", {"pattern": "def main", "include": "*.py"}, Workspace(tmp_path), config
    )
    assert r.ok
    assert "a.py" in r.output


def test_apply_patch(tmp_path, config):
    (tmp_path / "f.py").write_text("x = 1\ny = 2\n")
    r = execute_tool(
        "apply_patch",
        {"path": "f.py", "old_text": "x = 1", "new_text": "x = 10"},
        Workspace(tmp_path),
        config,
    )
    assert r.ok
    assert (tmp_path / "f.py").read_text() == "x = 10\ny = 2\n"


def test_apply_patch_missing_old_text(tmp_path, config):
    (tmp_path / "f.py").write_text("x = 1\n")
    r = execute_tool(
        "apply_patch",
        {"path": "f.py", "old_text": "zzz", "new_text": "yyy"},
        Workspace(tmp_path),
        config,
    )
    assert not r.ok
    assert (tmp_path / "f.py").read_text() == "x = 1\n"


def test_apply_patch_ambiguous(tmp_path, config):
    (tmp_path / "f.py").write_text("hi\nhi\n")
    r = execute_tool(
        "apply_patch",
        {"path": "f.py", "old_text": "hi", "new_text": "yo"},
        Workspace(tmp_path),
        config,
    )
    assert not r.ok
    assert "unique" in r.output.lower() or "matches" in r.output.lower()


def test_run_command_success(tmp_path, config):
    # Use single quotes inside the Python -c string: on Windows cmd, double
    # quotes are stripped and would break the embedded strings.
    code = (
        "import sys; print('hello-out'); print('hello-err', file=sys.stderr); "
        "sys.exit(3)"
    )
    r = execute_tool(
        "run_command", {"command": f'{sys.executable} -c "{code}"'}, Workspace(tmp_path), config
    )
    assert r.ok
    assert "hello-out" in r.output
    assert "hello-err" in r.output
    assert "exit code: 3" in r.output


def test_run_command_failure_nonzero(tmp_path, config):
    r = execute_tool(
        "run_command",
        {"command": f"{sys.executable} -c \"import sys; sys.exit(1)\""},
        Workspace(tmp_path),
        config,
    )
    assert r.ok or r.error  # nonzero exit is reported, not thrown


def test_run_command_timeout_is_a_failure(tmp_path, config):
    r = execute_tool(
        "run_command",
        {
            "command": f'{sys.executable} -c "import time; time.sleep(30)"',
            "timeout": 1,
        },
        Workspace(tmp_path),
        config,
    )
    assert not r.ok  # a timeout never counts as success
    assert "TIMED OUT" in r.output
    assert "timed out" in (r.note or "")


def test_workspace_escape_via_tool_blocked(tmp_path, config):
    # write_file with ../ traversal must be blocked and must not create anything
    outside = tmp_path.parent / "escape_should_not_exist.txt"
    r = execute_tool("write_file", {"path": "../escape_should_not_exist.txt", "content": "x"}, Workspace(tmp_path), config)
    assert not r.ok
    assert not outside.exists()
    assert "Workspace violation" in r.output


def test_unknown_tool_returns_failure(tmp_path, config):
    r = execute_tool("frobnicate", {}, Workspace(tmp_path), config)
    assert not r.ok
    assert "Unknown tool" in r.output


def test_missing_required_argument(tmp_path, config):
    r = execute_tool("read_file", {}, Workspace(tmp_path), config)
    assert not r.ok
    assert "path" in r.output


def test_validate_invalid_type_raises():
    with pytest.raises(ToolValidationError):
        validate_tool_call("read_file", {"path": 123})


def test_delete_file(tmp_path, config):
    ws = Workspace(tmp_path)
    (tmp_path / "gone.py").write_text("x = 1")
    r = execute_tool("delete_file", {"path": "gone.py"}, ws, config)
    assert r.ok
    assert not (tmp_path / "gone.py").exists()


def test_delete_file_missing_is_reported(tmp_path, config):
    ws = Workspace(tmp_path)
    r = execute_tool("delete_file", {"path": "missing.txt"}, ws, config)
    assert not r.ok  # deleting a file that never existed is a failure


def test_delete_file_refuses_root(tmp_path, config):
    ws = Workspace(tmp_path)
    r = execute_tool("delete_file", {"path": "."}, ws, config)
    assert not r.ok


def test_delete_file_refuses_vcs_metadata(tmp_path, config):
    ws = Workspace(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("x")
    r1 = execute_tool("delete_file", {"path": ".gitignore"}, ws, config)
    assert not r1.ok
    r2 = execute_tool("delete_file", {"path": ".git"}, ws, config)
    assert not r2.ok
    assert (tmp_path / ".gitignore").read_text() == "x"


def test_delete_file_refuses_directories(tmp_path, config):
    ws = Workspace(tmp_path)
    (tmp_path / "folder").mkdir()
    r = execute_tool("delete_file", {"path": "folder"}, ws, config)
    assert not r.ok


def test_move_file(tmp_path, config):
    ws = Workspace(tmp_path)
    (tmp_path / "a.txt").write_text("data")
    r = execute_tool("move_file", {"path": "a.txt", "destination": "b/c.txt"}, ws, config)
    assert r.ok
    assert not (tmp_path / "a.txt").exists()
    assert (tmp_path / "b" / "c.txt").read_text() == "data"


def test_move_file_into_directory(tmp_path, config):
    ws = Workspace(tmp_path)
    (tmp_path / "a.txt").write_text("data")
    (tmp_path / "src").mkdir()
    r = execute_tool("move_file", {"path": "a.txt", "destination": "src"}, ws, config)
    assert r.ok
    assert (tmp_path / "src" / "a.txt").exists()


def test_move_file_refuses_root(tmp_path, config):
    ws = Workspace(tmp_path)
    r = execute_tool("move_file", {"path": ".", "destination": "x"}, ws, config)
    assert not r.ok


def test_copy_file(tmp_path, config):
    ws = Workspace(tmp_path)
    (tmp_path / "src.txt").write_text("data")
    r = execute_tool("copy_file", {"path": "src.txt", "destination": "dst.txt"}, ws, config)
    assert r.ok
    assert (tmp_path / "dst.txt").read_text() == "data"
    assert (tmp_path / "src.txt").exists()


def test_set_plan(tmp_path, config):
    ws = Workspace(tmp_path)
    r = execute_tool("set_plan", {"goal": "g", "plan": ["one", "two"]}, ws, config)
    assert r.ok
    assert "recommended first step" in r.output or "one" in r.output


def test_set_plan_from_formatted_updates_step(tmp_path, config):
    ws = Workspace(tmp_path)
    r = execute_tool(
        "set_plan",
        {
            "plan": [
                "1. inspect",
                "2. implement feature X",
                "- [ ] design review",
                "4. run tests stub",
            ]
        },
        ws,
        config,
    )
    assert r.ok
    assert "implement feature X" in r.output


def test_inspect_environment_returns_facts(tmp_path, config):
    r = execute_tool("inspect_environment", {}, Workspace(tmp_path), config)
    assert r.ok
    assert "os:" in r.output
    assert "workspace:" in r.output
    assert sys.platform in r.output
