"""Command-line entry point for the coding agent (``risa``)."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import load_config
from .loop import run_agent
from .ollama import OllamaClient, OllamaError
from .workspace import Workspace, WorkspaceError


def _print_step(msg: str) -> None:
    print(msg, flush=True)


def _print_error(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="risa",
        description="RISARMS local autonomous coding agent (Ollama backend).",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Development task. If omitted, you are prompted for one.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Repository/workspace to operate on (default: current directory).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model to use (default: qwen2.5-coder:7b).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Ollama server URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum agent iterations (default: 50).",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Require approval before modifications and command execution (SAFE mode).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Execute valid tool calls automatically (default).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full tool output and reasoning.",
    )
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=None,
        help="Default timeout for run_command in seconds (default: 120).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify Ollama connectivity/model availability, then exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models installed on the Ollama server, then exit.",
    )
    parser.add_argument(
        "--version", action="version", version=f"risa {__version__}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.safe and args.auto:
            _print_error("Cannot use --safe and --auto together.")
            return 2
        overrides: dict = {
            "workspace": args.workspace or ".",
        }
        for key, value in (
            ("model", args.model),
            ("ollama_base_url", args.base_url),
            ("max_iterations", args.max_iterations),
            ("command_timeout", args.command_timeout),
        ):
            if value is not None:
                overrides[key] = value
        if args.verbose:
            overrides["verbose"] = True
        if args.safe:
            overrides["mode"] = "SAFE"
        elif args.auto:
            overrides["mode"] = "AUTO"
        config = load_config(**overrides)
    except (ValueError, WorkspaceError) as exc:
        _print_error(str(exc))
        return 2

    client = OllamaClient(base_url=config.ollama_base_url, model=config.model)

    # -- standalone preflight commands -------------------------------------
    if args.list_models:
        return _cmd_list_models(client)
    if args.check:
        return _cmd_check(client)

    # -- preflight connectivity --------------------------------------------
    try:
        client.check_connectivity(timeout=5)
    except Exception as exc:
        _print_error(
            f"Cannot reach Ollama at {config.ollama_base_url}. "
            f"Start it with `ollama serve`.\nDetails: {exc}"
        )
        return 1
    try:
        available = client.is_model_available(config.model)
    except Exception as exc:
        _print_error(f"Could not query Ollama models: {exc}")
        return 1
    if not available:
        try:
            installed = client.list_models()
        except Exception:
            installed = []
        hint = ", ".join(installed) if installed else "(none installed)"
        _print_error(
            f"Model '{config.model}' is not installed on the Ollama server.\n"
            f"Installed models: {hint}\n"
            f"Pull it with: ollama pull {config.model}\n"
            f"Or select an installed model with OLLAMA_MODEL or --model."
        )
        return 1

    _print_step(f"risa v{__version__} — Ollama model: {config.model}")
    _print_step(f"Workspace: {config.workspace}")

    # -- task acquisition ---------------------------------------------------
    task = args.task
    if task is None or not task.strip():
        try:
            task = input("Task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo task given; exiting.")
            return 0
        if not task:
            return 0

    try:
        Workspace(config.workspace)
    except WorkspaceError as exc:
        _print_error(str(exc))
        return 2

    try:
        result = run_agent(config, client, task, log=_print_step)
    except KeyboardInterrupt:
        _print_step("\nInterrupted.")
        return 130
    except OllamaError as exc:
        _print_step(f"\nSTATUS: FATAL")
        _print_error(str(exc))
        return 1

    print(f"\nSTATUS: {result.status.upper()}")
    if result.iterations:
        print(f"Iterations: {result.iterations}")
    if result.summary:
        print(f"Summary: {result.summary}")
    if result.error:
        _print_error(result.error)

    return 0 if result.is_complete else 1


def _cmd_list_models(client: OllamaClient) -> int:
    try:
        models = client.list_models(timeout=10)
    except Exception as exc:
        _print_error(f"Could not reach Ollama: {exc}")
        return 1
    print("Installed models:")
    for name in models:
        print(f"  {name}")
    return 0


def _cmd_check(client: OllamaClient) -> int:
    try:
        client.check_connectivity(timeout=5)
        models = client.list_models(timeout=10)
    except Exception as exc:
        _print_error(f"Ollama check failed: {exc}")
        return 1
    print("Ollama: OK")
    print(f"Model {client.model!r}: "
          f"{'available' if client.model in models else 'NOT INSTALLED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())