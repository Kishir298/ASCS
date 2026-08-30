"""Tests for workspace containment / path-traversal security."""

from __future__ import annotations

import os

import pytest

from agent.workspace import Workspace, WorkspaceError, should_ignore


def make_ws(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.txt").write_text("inner")
    return Workspace(tmp_path)


def test_resolve_relative(tmp_path):
    ws = make_ws(tmp_path)
    assert ws.resolve("sub/inner.txt") == (tmp_path / "sub" / "inner.txt").resolve()
    assert ws.resolve(".") == tmp_path.resolve()


def test_resolve_absolute_within(tmp_path):
    ws = make_ws(tmp_path)
    target = tmp_path / "sub" / "inner.txt"
    assert ws.resolve(str(target)) == target.resolve()


def test_resolve_absolutes_resolve_against_root(tmp_path):
    ws = make_ws(tmp_path)
    assert ws.resolve("/whatever/../sub/inner.txt") == (
        tmp_path / "sub" / "inner.txt"
    ).resolve()


def test_parent_traversal_rejected(tmp_path):
    ws = make_ws(tmp_path)
    for bad in ("../outside.txt", "sub/../../outside.txt", ".."):
        with pytest.raises(WorkspaceError):
            ws.resolve(bad)


def test_absolute_outside_rejected(tmp_path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    ws = make_ws(tmp_path)
    target = outside / "secret.txt"
    target.write_text("secret")
    with pytest.raises(WorkspaceError):
        ws.resolve(str(target))


def test_is_inside(tmp_path):
    ws = make_ws(tmp_path)
    assert ws.is_inside("sub/inner.txt")
    assert ws.is_inside(".")
    assert not ws.is_inside("../outside.txt")
    assert not ws.is_inside(str(tmp_path.parent / "x"))


@pytest.mark.skipif(os.name != "nt", reason="Windows-only junction escape test")
def test_junction_escape_rejected(tmp_path):
    # A junction pointing outside must be treated as an escape.
    outside = tmp_path.parent / ("outside_" + tmp_path.name)
    outside.mkdir(exist_ok=True)
    outside_file = outside / "secret.txt"
    outside_file.write_text("secret")
    junction = tmp_path / "link"
    try:
        import subprocess

        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
            check=True,
            capture_output=True,
        )
    except Exception:
        pytest.skip("could not create junction (may need extra privileges)")
    try:
        ws = Workspace(tmp_path)
        with pytest.raises(WorkspaceError):
            ws.resolve("link/secret.txt")
    finally:
        from agent.workspace import _remove_link
        _remove_link(junction)
        if outside.exists():
            import shutil
            shutil.rmtree(outside, ignore_errors=True)


def test_ignore_rules():
    assert should_ignore(".git")
    assert should_ignore(".venv")
    assert should_ignore("__pycache__")
    assert should_ignore("node_modules")
    assert should_ignore("some.egg-info")
    assert should_ignore("SOME.EGG-INFO")  # case-insensitive suffix
    assert not should_ignore("main.py")
    assert not should_ignore(".gitignore")  # file, not the dir


def test_iter_files_skips_ignored(tmp_path):
    (tmp_path / "a.py").write_text("a")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "skip.py").write_text("skip")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    ws = Workspace(tmp_path)
    names = sorted(p.name for p in Workspace.iter_files(tmp_path))
    assert names == ["a.py", "b.txt"]


def test_root_must_be_directory(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x")
    with pytest.raises(WorkspaceError):
        Workspace(f)


def test_nonexistent_root_raises(tmp_path):
    with pytest.raises(WorkspaceError):
        Workspace(tmp_path / "nope")
