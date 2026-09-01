"""Tool definitions, validation, and execution.

Every tool follows the same contract: ``{tool, description, arguments_schema,
validate, run}``. The ``execute_tool`` dispatcher validates the call and turns
any failure into a ``ToolResult`` with ``ok=False`` so the model can adapt.
"""

from __future__ import annotations

import fnmatch
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .models import Plan, ToolResult, truncate
from .workspace import Workspace, WorkspaceError, should_ignore

TRUNCATION_MARKER = "\n... [TRUNCATED by coding-agent; use targeted tools/arguments to see the rest]"


class ToolValidationError(Exception):
    """Raised when a tool call's arguments are invalid."""


def truncate_env(text: str, max_chars: int) -> str:
    return truncate(text, max_chars, marker=TRUNCATION_MARKER)


# -- argument helpers ------------------------------------------------------


def _require(obj: dict[str, Any], key: str, expected: type) -> Any:
    if key not in obj or obj[key] is None:
        raise ToolValidationError(f"Missing required argument: {key!r}")
    value = obj[key]
    if expected is Any:
        return value  # presence-checked; no type constraint
    if not isinstance(value, expected):
        raise ToolValidationError(
            f"Argument {key!r} must be of type {expected.__name__}, got {type(value).__name__}."
        )
    return value


def _opt_str(obj: dict[str, Any], key: str, default: str) -> str:
    if key not in obj or obj[key] is None:
        return default
    value = obj[key]
    if not isinstance(value, str):
        raise ToolValidationError(f"Argument {key!r} must be a string.")
    return value


def _opt_int(obj: dict[str, Any], key: str, default: int) -> int:
    if key not in obj or obj[key] is None:
        return default
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolValidationError(f"Argument {key!r} must be an integer.")
    if value < 0:
        raise ToolValidationError(f"Argument {key!r} must be non-negative.")
    return value


def _opt_bool(obj: dict[str, Any], key: str, default: bool) -> bool:
    if key not in obj or obj[key] is None:
        return default
    value = obj[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ToolValidationError(f"Argument {key!r} must be a boolean.")


def _opt_any(obj: dict[str, Any], key: str, default: Any = None) -> Any:
    """Pass-through for heterogeneous arguments (lists/dicts/strings)."""
    if key not in obj or obj[key] is None:
        return default
    return obj[key]


# -- individual tools ------------------------------------------------------


def _list_directory(args: dict[str, Any], ws: Workspace, cfg: Any) -> ToolResult:
    path = _opt_str(args, "path", ".")
    recursive = _opt_bool(args, "recursive", False)
    target = ws.resolve(path)
    if not target.is_dir():
        raise ToolValidationError(f"Not a directory: {path}")

    entries: list[str] = []
    root_cmp = ws.root
    try:
        rel = target.relative_to(root_cmp)
        prefix = "" if str(rel) == "." else str(rel)
    except ValueError:
        prefix = str(target)

    def line_for(path_: Path) -> str:
        if path_.is_dir():
            return f"[dir]  {path_.name}/"
        try:
            size = path_.stat().st_size
        except OSError:
            size = -1
        return f"[file] {path_.name} ({size} bytes)"

    if recursive:
        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = [d for d in dirnames if not should_ignore(d)]
            for d in sorted(dirnames):
                rel_d = Path(dirpath).relative_to(root_cmp) / d
                entries.append(f"[dir]  {rel_d}/")
            for name in sorted(filenames):
                if should_ignore(name):
                    continue
                p = Path(dirpath) / name
                rel_f = p.relative_to(root_cmp)
                try:
                    size = p.stat().st_size
                except OSError:
                    size = -1
                entries.append(f"[file] {rel_f} ({size} bytes)")
    else:
        names = sorted(
            os.listdir(target),
            key=lambda n: (not (target / n).is_dir(), n.lower()),
        )
        for name in names:
            if should_ignore(name):
                continue
            entries.append(line_for(target / name))

    if not entries:
        return ToolResult("list_directory", "(directory is empty)")

    header = f"Listing {'recursive ' if recursive else ''}of {prefix or '.'} ({len(entries)} entries):"
    body = "\n".join(entries[:1000])
    if len(entries) > 1000:
        body += f"\n... {len(entries) - 1000} more entries"
    return ToolResult("list_directory", header + "\n" + body)


def _read_file(args: dict[str, Any], ws: Workspace, cfg: Any) -> ToolResult:
    path = _require(args, "path", str)
    target = ws.resolve(path)
    if not target.exists():
        raise ToolValidationError(f"File does not exist: {path}")
    if not target.is_file():
        raise ToolValidationError(f"Not a regular file: {path}")

    if target.suffix.lower() in _BINARY_SUFFIXES:
        return ToolResult(
            "read_file",
            f"{path} looks like a binary file (extension {target.suffix}). "
            "Use run_command if you need to inspect it.",
            ok=False,
        )

    raw = target.read_bytes()
    if _looks_binary(raw):
        return ToolResult(
            "read_file",
            f"{path} contains binary data and cannot be read as text. "
            "Use run_command if you need to inspect it.",
            ok=False,
        )

    text, encoding = _decode_with_fallback(raw)
    lines = text.splitlines()

    start_line = _opt_int(args, "start_line", 0)
    end_line = _opt_int(args, "end_line", 0)
    note = ""
    if start_line or end_line:
        start = max(1, start_line)
        end = end_line if end_line else len(lines)
        end = min(end, len(lines))
        if start > len(lines):
            return ToolResult("read_file", f"{path} has only {len(lines)} lines.")
        selected = lines[start - 1:end]
        note = f" (lines {start}-{end} of {len(lines)})"
    else:
        max_lines = 2000
        if len(lines) > max_lines:
            selected = lines[:max_lines]
            note = f" (first {max_lines} of {len(lines)} lines; use start_line/end_line to read more)"
        else:
            selected = lines

    width = len(str(len(lines)))
    numbered = "\n".join(f"{i+1:>{width}}| {ln}" for i, ln in enumerate(selected))
    enc_note = f" [decoded as {encoding}]" if encoding != "utf-8" else ""
    output = f"{path}{note}{enc_note}:\n{numbered}"
    return ToolResult("read_file", truncate_env(output, cfg.max_output_chars))


def _search_files(args: dict[str, Any], ws: Workspace, cfg: Any) -> ToolResult:
    pattern_raw = _require(args, "pattern", str)
    path = _opt_str(args, "path", ".")
    include = _opt_str(args, "include", "") or None
    case_sensitive = _opt_bool(args, "case_sensitive", False)
    max_results = _opt_int(args, "max_results", 100) or 100
    max_results = min(max_results, 500)

    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern_raw, flags)
    except re.error as exc:
        raise ToolValidationError(f"Invalid regex {pattern_raw!r}: {exc}") from exc

    root = ws.resolve(path)
    if not root.is_dir():
        raise ToolValidationError(f"Not a directory: {path}")

    matches: list[str] = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_ignore(d)]
        for name in filenames:
            if should_ignore(name):
                continue
            if include and not fnmatch.fnmatch(name, include):
                continue
            full = Path(dirpath) / name
            total += 1
            try:
                with full.open("r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        if regex.search(line.rstrip("\n")):
                            rel = full.relative_to(ws.root)
                            snippet = line.rstrip("\n")
                            if len(snippet) > 200:
                                snippet = snippet[:197] + "..."
                            matches.append(
                                f"{rel}:{lineno}: {snippet}"
                            )
                            if len(matches) >= max_results:
                                break
            except (OSError, UnicodeError):
                continue
            if len(matches) >= max_results:
                break
        if len(matches) >= max_results:
            break

    if not matches:
        return ToolResult(
            "search_files",
            f"No matches for pattern {pattern_raw!r} under {path}.",
        )
    body = "\n".join(matches)
    if len(matches) >= max_results:
        body += f"\n... stopped at {max_results} matches (max_results limit)."
    return ToolResult(
        "search_files",
        f"{len(matches)} match(es) for {pattern_raw!r} (scanned {total} files):\n{body}",
    )


def _write_file(args: dict[str, Any], ws: Workspace, cfg: Any) -> ToolResult:
    path = _require(args, "path", str)
    content = _require(args, "content", str)
    target = ws.resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    data = content.encode("utf-8")
    tmp = target.with_name(target.name + ".risa_tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    return ToolResult(
        "write_file",
        f"Wrote {len(data)} bytes to {path}",
        note="file created/replaced",
    )


def _apply_patch(args: dict[str, Any], ws: Workspace, cfg: Any) -> ToolResult:
    path = _require(args, "path", str)
    old_text = _require(args, "old_text", str)
    new_text = _require(args, "new_text", str)
    target = ws.resolve(path)
    if not target.is_file():
        raise ToolValidationError(f"Not a regular file: {path}")

    text, _ = _decode_with_fallback(target.read_bytes())
    count = text.count(old_text)
    if count == 0:
        return ToolResult(
            "apply_patch",
            f"old_text was not found in {path}. Read the file first and include "
            "exact existing content (including full lines) in old_text.",
            ok=False,
        )
    if count > 1:
        return ToolResult(
            "apply_patch",
            f"old_text matches {count} times in {path}. Include more surrounding "
            "context to make the replacement unique.",
            ok=False,
        )
    updated = text.replace(old_text, new_text, 1)
    tmp = target.with_name(target.name + ".risa_tmp")
    tmp.write_bytes(updated.encode("utf-8"))
    os.replace(tmp, target)
    return ToolResult(
        "apply_patch",
        f"Applied patch to {path} (replaced 1 occurrence).",
        note="file modified",
    )


def _delete_file(args: dict[str, Any], ws: Workspace, cfg: Any) -> ToolResult:
    path = _require(args, "path", str)
    target = ws.resolve(path)
    if target == ws.root:
        raise ToolValidationError("Refusing to delete the workspace root itself.")
    if not target.exists() and not target.is_symlink():
        raise ToolValidationError(f"Path does not exist: {path}")
    if target.is_dir() and not target.is_symlink():
        raise ToolValidationError(
            f"Refusing to delete directory {path!r} with delete_file; "
            "remove directories via run_command (e.g. Remove-Item -Recurse) "
            "with explicit operator intent."
        )
    try:
        rel = target.relative_to(ws.root)
    except ValueError:
        raise ToolValidationError(f"Path resolves outside the workspace: {path}")
    if any(part in (".git", ".github") for part in rel.parts):
        raise ToolValidationError(
            f"Refusing to delete version-control metadata at {path!r}."
        )
    if rel.name in (".gitignore",):
        raise ToolValidationError(
            f"Refusing to delete {rel.name!r}; remove it via run_command if that "
            "is genuinely required."
        )
    if target.is_symlink():
        from .workspace import _remove_link
        _remove_link(target)
        return ToolResult(
            "delete_file", f"Removed link {path}", note="link deleted"
        )
    target.unlink()
    return ToolResult("delete_file", f"Deleted {path}", note="file deleted")


def _move_file(args: dict[str, Any], ws: Workspace, cfg: Any) -> ToolResult:
    path = _require(args, "path", str)
    destination = _require(args, "destination", str)
    src = ws.resolve(path)
    dst = ws.resolve(destination)
    if src == ws.root:
        raise ToolValidationError("Refusing to move the workspace root itself.")
    if not src.exists() and not src.is_symlink():
        raise ToolValidationError(f"Source does not exist: {path}")
    if dst.exists():
        if not dst.is_dir():
            raise ToolValidationError(
                f"Destination already exists and is not a directory: {destination}"
            )
        dst = dst / src.name
        if dst.exists():
            raise ToolValidationError(
                f"Destination already exists: {dst.relative_to(ws.root)}"
            )
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)
    rel = dst.relative_to(ws.root)
    return ToolResult("move_file", f"Moved {path} to {rel}", note="file moved")


def _copy_file(args: dict[str, Any], ws: Workspace, cfg: Any) -> ToolResult:
    path = _require(args, "path", str)
    destination = _require(args, "destination", str)
    src = ws.resolve(path)
    dst = ws.resolve(destination)
    if not src.is_file():
        raise ToolValidationError(f"Source is not a regular file: {path}")
    if dst.exists() and not dst.is_dir():
        raise ToolValidationError(
            f"Destination already exists: {destination}"
        )
    if dst.is_dir():
        dst = dst / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(src, dst)
    rel = dst.relative_to(ws.root)
    return ToolResult(
        "copy_file",
        f"Copied {path} to {rel} ({dst.stat().st_size} bytes)",
        note="file copied",
    )


def _set_plan(args: dict[str, Any], ws: Workspace, cfg: Any) -> ToolResult:
    value = _opt_any(args, "plan")
    goal = _opt_str(args, "goal", "")
    plan = Plan.from_value(value)
    return ToolResult(
        "set_plan",
        plan.to_text(),
        note="plan recorded",
    )


def _inspect_environment(args: dict[str, Any], ws: Workspace, cfg: Any) -> ToolResult:
    """Report host/platform facts so the model never guesses workflows."""
    def _which(name: str) -> str:
        found = shutil.which(name)
        return found or "(not on PATH)"

    lines = [
        f"os: {os.name} ({sys.platform})",
        f"platform: {platform.system()} {platform.release()} "
        f"({platform.machine()})",
        f"shell for run_command: "
        f"{os.environ.get('COMSPEC') or '/bin/sh'} (default on this OS)",
        f"python (this agent): {sys.executable}",
        f"python on PATH: {_which('python')}",
        f"py launcher: {_which('py')}",
        f"pip: {_which('pip')}",
        f"pytest: {_which('pytest')}",
        f"git: {_which('git')}",
        f"workspace: {ws.root}",
        f"pathsep: {os.pathsep}",
        "hint: run_command already makes the running interpreter's directory "
        "available on PATH, so `python`, `pip`, and `pytest` should resolve.",
    ]
    body = "\n".join(lines)
    if os.name == "nt":
        body += (
            "\nWindows tips:\n"
            "- Commands run through cmd.exe; `dir`, `type`, `copy`, `move`, "
            "`del`, `findstr` work.\n"
            "- Prefer `python -m pytest`, `python -m pip ...` so the right "
            "interpreter is used.\n"
            "- For PowerShell idioms, wrap them: "
            '`powershell -NoProfile -Command "Get-ChildItem"`.\n'
            "- venv activation on Windows: `env\\Scripts\\activate.bat`; you "
            "can also call its python directly."
        )
    return ToolResult("inspect_environment", body)


def _run_command(args: dict[str, Any], ws: Workspace, cfg: Any) -> ToolResult:
    command = _require(args, "command", str)
    if not command.strip():
        raise ToolValidationError("command must not be empty.")
    timeout = _opt_int(args, "timeout", 0) or cfg.command_timeout

    proc, stdout, stderr, rc, timed_out, killed = _execute_process(
        command, ws.root, timeout
    )
    if killed:
        raise KeyboardInterrupt  # handled by the loop as a user interrupt
    status_line = "TIMED OUT" if timed_out else str(rc)
    output = (
        f"exit code: {status_line}\n"
        f"--- stdout ---\n{stdout}\n"
        f"--- stderr ---\n{stderr}"
    )
    # A timeout never counts as success: the model should treat it as a
    # failure to inspect and adapt to, not a completed step.
    return ToolResult(
        "run_command",
        truncate_env(output, cfg.max_output_chars),
        ok=not timed_out,
        note=(f"timed out after {timeout}s" if timed_out else f"exit code {rc}"),
    )


def _terminate_tree(proc: subprocess.Popen) -> None:
    """Force-kill ``proc`` and (on Windows) its descendant process tree.

    A plain ``proc.kill()`` on Windows only terminates the intermediate
    ``cmd.exe``; grandchildren (the actual command) would keep running. Using
    ``taskkill /T /F`` kills the whole tree.
    """
    if os.name == "nt" and proc.poll() is None:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
            )
            return
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _execute_process(
    command: str, cwd: Path, timeout: int
) -> tuple[subprocess.Popen, str, str, int | None, bool, bool]:
    """Run a shell command capturing output, honoring timeout and Ctrl+C.

    The directory containing the running interpreter is prepended to PATH so
    ``python``/``pip``/``pytest`` resolve even when the system PATH lacks them
    (common on Windows). No personal/user paths are hard-coded.
    """
    env = os.environ.copy()
    # Use the *unresolved* interpreter directory. ``sys.executable`` inside a
    # virtualenv is a symlink (e.g. .venv/bin/python -> system python); resolving
    # it would point at the framework interpreter's dir, which lacks the venv's
    # ``python``/``pip``/``pytest`` shims and so breaks ``python`` resolution.
    # Prepend both the venv dir (provides the shims) and the resolved dir.
    bin_dir = str(Path(sys.executable).parent)
    resolved_bin = str(Path(sys.executable).resolve().parent)
    existing = env.get("PATH", "")
    extra = os.pathsep.join(
        d for d in (bin_dir, resolved_bin) if d not in ("",) and d not in existing.split(os.pathsep)
    )
    env["PATH"] = (extra + os.pathsep if extra else "") + existing
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "cwd": str(cwd),
        "shell": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(command, **kwargs)
    start = time.monotonic()
    timed_out = False
    killed = False
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - start > timeout:
                    timed_out = True
                    _terminate_tree(proc)
                    stdout, stderr = proc.communicate()
                    break
    except KeyboardInterrupt:
        killed = True
        _terminate_tree(proc)
        try:
            proc.communicate()
        except Exception:  # pragma: no cover - defensive
            pass
        return proc, "interrupted", "process killed by user", None, False, True
    return proc, stdout if stdout else "", stderr if stderr else "", proc.returncode, timed_out, False


def _git_command(
    name: str, args_list: list[str], ws: Workspace, timeout: int = 60
) -> ToolResult:
    try:
        proc = subprocess.run(
            ["git", *args_list],
            cwd=str(ws.root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError:
        return ToolResult(name, "git is not installed or not on PATH.", ok=False)
    except subprocess.TimeoutExpired:
        return ToolResult(name, f"git {name} timed out after {timeout}s.", ok=False)
    combined = (proc.stdout or "") + (proc.stderr or "")
    rc = proc.returncode
    if rc != 0:
        # Working in a directory that is not a git repo is not an error: the
        # agent should proceed, not treat it as a retryable failure.
        lowered = combined.strip().lower()
        if rc == 128 and "not a git repository" in lowered:
            return ToolResult(name, "(not a git repository)", note="exit code 128")
        return ToolResult(name, combined.strip() or f"git exited with {rc}", ok=False)
    body = combined.strip() or {"git_status": "(clean working tree)", "git_diff": "(no changes)"}[name]
    return ToolResult(name, body, note=f"exit code 0")


def _git_status(args: dict[str, Any], ws: Workspace, cfg: Any) -> ToolResult:
    return _git_command("git_status", ["status", "--short"], ws)


def _git_diff(args: dict[str, Any], ws: Workspace, cfg: Any) -> ToolResult:
    staged = _opt_bool(args, "staged", False)
    return _git_command(
        "git_diff", ["diff", "--cached"] if staged else ["diff"], ws
    )


# -- registry ---------------------------------------------------------------

ToolHandler = Callable[[dict[str, Any], Workspace, Any], ToolResult]


class ToolSpec:
    def __init__(
        self,
        name: str,
        summary: str,
        arguments_schema: dict[str, dict[str, Any]],
        handler: ToolHandler,
        example: str,
    ) -> None:
        self.name = name
        self.summary = summary
        self.arguments_schema = arguments_schema
        self.handler = handler
        self.example = example

    def prompt_text(self) -> str:
        parts = [f"- {self.name}: {self.summary}"]
        parts.append(f"  arguments: {json.dumps(self.arguments_schema, sort_keys=True)}")
        parts.append(f"  example: {json.dumps(self.example, sort_keys=True)}")
        return "\n".join(parts)


TOOL_SPECS: dict[str, ToolSpec] = {}


def register_tool(spec: ToolSpec) -> ToolSpec:
    TOOL_SPECS[spec.name] = spec
    return spec


register_tool(
    ToolSpec(
        name="list_directory",
        summary="List files and directories in the workspace (ignores generated/VCS dirs).",
        arguments_schema={
            "path": {"type": "string", "description": "Directory to list; default '.'.", "required": False, "default": "."},
            "recursive": {"type": "boolean", "description": "List recursively; default false.", "required": False, "default": False},
        },
        handler=_list_directory,
        example={"tool": "list_directory", "arguments": {"path": "."}},
    )
)
register_tool(
    ToolSpec(
        name="read_file",
        summary="Read a text file inside the workspace with line numbers.",
        arguments_schema={
            "path": {"type": "string", "description": "Path relative to workspace.", "required": True},
            "start_line": {"type": "integer", "description": "First line (1-based) to read; optional.", "required": False},
            "end_line": {"type": "integer", "description": "Last line (inclusive) to read; optional.", "required": False},
        },
        handler=_read_file,
        example={"tool": "read_file", "arguments": {"path": "src/example.py", "start_line": 1, "end_line": 40}},
    )
)
register_tool(
    ToolSpec(
        name="search_files",
        summary="Recursively search file contents in the workspace with a regular expression.",
        arguments_schema={
            "pattern": {"type": "string", "description": "Regular expression to search for.", "required": True},
            "path": {"type": "string", "description": "Directory to search; default '.'.", "required": False, "default": "."},
            "include": {"type": "string", "description": "Optional glob filter on filenames, e.g. '*.py'.", "required": False},
            "max_results": {"type": "integer", "description": "Maximum matches; default 100.", "required": False, "default": 100},
            "case_sensitive": {"type": "boolean", "description": "Match case; default false.", "required": False, "default": False},
        },
        handler=_search_files,
        example={"tool": "search_files", "arguments": {"pattern": "def main", "include": "*.py"}},
    )
)
register_tool(
    ToolSpec(
        name="write_file",
        summary="Create a new file or replace an existing file inside the workspace.",
        arguments_schema={
            "path": {"type": "string", "description": "Path relative to workspace; parent dirs are created.", "required": True},
            "content": {"type": "string", "description": "Full new file content.", "required": True},
        },
        handler=_write_file,
        example={"tool": "write_file", "arguments": {"path": "hello.py", "content": "print('hi')\n"}},
    )
)
register_tool(
    ToolSpec(
        name="apply_patch",
        summary="Replace an exact existing substring in a file (single, unambiguous occurrence).",
        arguments_schema={
            "path": {"type": "string", "description": "Path relative to workspace.", "required": True},
            "old_text": {"type": "string", "description": "Exact existing text to replace (must appear exactly once).", "required": True},
            "new_text": {"type": "string", "description": "Replacement text.", "required": True},
        },
        handler=_apply_patch,
        example={"tool": "apply_patch", "arguments": {"path": "src/app.py", "old_text": "return 1", "new_text": "return 2"}},
    )
)
register_tool(
    ToolSpec(
        name="delete_file",
        summary="Delete one regular file (or symlink) inside the workspace. Refuses directories, the workspace root, and VCS metadata.",
        arguments_schema={
            "path": {"type": "string", "description": "Path relative to workspace to delete.", "required": True},
        },
        handler=_delete_file,
        example={"tool": "delete_file", "arguments": {"path": "stale.txt"}},
    )
)
register_tool(
    ToolSpec(
        name="move_file",
        summary="Move a file (or symlink) to a new path inside the workspace.",
        arguments_schema={
            "path": {"type": "string", "description": "Source path relative to workspace.", "required": True},
            "destination": {"type": "string", "description": "Destination path relative to workspace; may be an existing directory.", "required": True},
        },
        handler=_move_file,
        example={"tool": "move_file", "arguments": {"path": "a.py", "destination": "src/a.py"}},
    )
)
register_tool(
    ToolSpec(
        name="copy_file",
        summary="Copy a regular file to a new path inside the workspace.",
        arguments_schema={
            "path": {"type": "string", "description": "Source path relative to workspace.", "required": True},
            "destination": {"type": "string", "description": "Destination path relative to workspace; may be an existing directory.", "required": True},
        },
        handler=_copy_file,
        example={"tool": "copy_file", "arguments": {"path": "a.py", "destination": "src/a.py"}},
    )
)
register_tool(
    ToolSpec(
        name="set_plan",
        summary="Record the structured implementation plan for the task. The plan is shown to the operator and drives BUILD execution.",
        arguments_schema={
            "plan": {"type": "any", "description": "Ordered list of plan steps. Each item is a string or {\"step\": ..., \"detail\": ...}. May also be a multi-line string.", "required": True},
            "goal": {"type": "string", "description": "One-sentence statement of the goal; optional.", "required": False},
        },
        handler=_set_plan,
        example={"tool": "set_plan", "arguments": {"goal": "Add a CLI flag", "plan": ["Inspect CLI parser", "Add the flag", "Run tests"]}},
    )
)
register_tool(
    ToolSpec(
        name="run_command",
        summary="Run a development command from the workspace root (pytest, python, git, ruff, ...).",
        arguments_schema={
            "command": {"type": "string", "description": "Shell command to run.", "required": True},
            "timeout": {"type": "integer", "description": "Override timeout in seconds; default from config.", "required": False},
        },
        handler=_run_command,
        example={"tool": "run_command", "arguments": {"command": "pytest", "timeout": 120}},
    )
)
register_tool(
    ToolSpec(
        name="inspect_environment",
        summary="Report host/platform facts: OS, shell, python/pip/pytest/git availability, workspace root.",
        arguments_schema={},
        handler=_inspect_environment,
        example={"tool": "inspect_environment", "arguments": {}},
    )
)
register_tool(
    ToolSpec(
        name="git_status",
        summary="Show the git working tree status of the workspace ('git status --short').",
        arguments_schema={},
        handler=_git_status,
        example={"tool": "git_status", "arguments": {}},
    )
)
register_tool(
    ToolSpec(
        name="git_diff",
        summary="Show uncommitted changes in the workspace ('git diff', or '--cached' for staged).",
        arguments_schema={
            "staged": {"type": "boolean", "description": "Show staged diff instead; default false.", "required": False, "default": False},
        },
        handler=_git_diff,
        example={"tool": "git_diff", "arguments": {"staged": True}},
    )
)


def get_tool_spec(name: str) -> ToolSpec | None:
    return TOOL_SPECS.get(name)


def validate_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate name + args against the registry. Raises ToolValidationError."""
    if not isinstance(name, str):
        raise ToolValidationError(f"Tool name must be a string, got {name!r}.")
    spec = TOOL_SPECS.get(name)
    if spec is None:
        valid = ", ".join(sorted(TOOL_SPECS)) or "none"
        raise ToolValidationError(
            f"Unknown tool: {name!r}. Valid tools: {valid}."
        )
    if not isinstance(arguments, dict):
        raise ToolValidationError(
            f"Tool {name!r} arguments must be a JSON object."
        )
    return _validate_args(spec, arguments)


def _validate_args(spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    schema = spec.arguments_schema
    checked: dict[str, Any] = {}
    for key, meta in schema.items():
        if meta.get("required"):
            _require(arguments, key, str if meta["type"] == "string" else Any)
        if meta["type"] == "string":
            checked[key] = _opt_str(arguments, key, meta.get("default", ""))
        elif meta["type"] == "integer":
            checked[key] = _opt_int(arguments, key, meta.get("default", 0))
        elif meta["type"] == "boolean":
            checked[key] = _opt_bool(arguments, key, meta.get("default", False))
        elif meta["type"] == "any":
            checked[key] = _opt_any(arguments, key, meta.get("default"))
    extra = set(arguments) - set(schema)
    if extra:
        unknown = ", ".join(sorted(extra))
        return {**checked, "_warnings": f"Ignored unknown arguments: {unknown}"}
    return checked


def execute_tool(
    name: str, arguments: dict[str, Any], ws: Workspace, cfg: Any
) -> ToolResult:
    """Validate and run a tool call. Never raises for logical failures."""
    try:
        validated = validate_tool_call(name, arguments)
    except ToolValidationError as exc:
        return ToolResult(name, str(exc), ok=False)
    warnings = validated.pop("_warnings", None)
    spec = TOOL_SPECS[name]
    try:
        result = spec.handler(validated, ws, cfg)
    except ToolValidationError as exc:
        return ToolResult(name, str(exc), ok=False)
    except WorkspaceError as exc:
        return ToolResult(name, f"Workspace violation: {exc}", ok=False)
    except OSError as exc:
        return ToolResult(name, f"OS error: {exc}", ok=False)
    except Exception as exc:  # defensive: never crash the loop on tool bugs
        return ToolResult(name, f"Unexpected {type(exc).__name__}: {exc}", ok=False)
    if warnings:
        result.output = f"{result.output}\n{warnings}"
    return result


def tool_schema_text(names: list[str] | None = None) -> str:
    """Render the tool reference for the system prompt.

    Renders every registered tool when ``names`` is None, otherwise only the
    named tools (which must exist in the registry).
    """
    if names is None:
        specs = TOOL_SPECS.values()
    else:
        specs = [TOOL_SPECS[name] for name in names]
    return "\n".join(spec.prompt_text() for spec in specs)


# -- binary detection -------------------------------------------------------

_BINARY_SUFFIXES = {
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".lib", ".obj",
    ".pyc", ".pyo", ".pyd", ".whl", ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".7z", ".rar", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".db",
    ".sqlite", ".sqlite3",
}


def _looks_binary(raw: bytes) -> bool:
    if not raw:
        return False
    sample = raw[:8192]
    if b"\x00" in sample:
        return True
    return False


def _decode_with_fallback(raw: bytes) -> tuple[str, str]:
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace"), "latin-1"


__all__ = [
    "TOOL_SPECS",
    "ToolSpec",
    "ToolValidationError",
    "get_tool_spec",
    "validate_tool_call",
    "execute_tool",
    "tool_schema_text",
    "TRUNCATION_MARKER",
    "truncate_env",
]