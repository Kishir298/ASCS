"""Lightweight CLI tests that avoid needing a live Ollama server."""

from __future__ import annotations

from agent.main import build_parser, main


def test_parser_has_expected_flags():
    p = build_parser()
    argv = [
        "--workspace",
        ".",
        "--model",
        "m",
        "--base-url",
        "u",
        "--max-iterations",
        "5",
        "--verbose",
        "--safe",
        "do the task",
    ]
    ns = p.parse_args(argv)
    assert ns.workspace == "."
    assert ns.model == "m"
    assert ns.base_url == "u"
    assert ns.max_iterations == 5
    assert ns.verbose is True
    assert ns.safe is True
    assert ns.task == "do the task"


def test_safe_and_auto_conflict_exits_2(capsys):
    rc = main(["--safe", "--auto", "--workspace", ".", "task"])
    assert rc == 2
    out = capsys.readouterr().err
    assert "Cannot use --safe and --auto" in out


def test_version_flag(capsys):
    with __import__("pytest").raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "risa" in out


def test_unknown_argument_is_error():
    import pytest

    with pytest.raises(SystemExit):
        main(["--definitely-not-a-flag"])
