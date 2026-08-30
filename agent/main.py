"""Command-line entry point for A.S.C.S. (``risa``).

Subcommands/behaviors:
    risa [OPTIONS] [TASK]         run a one-shot agent session
    risa --ui [OPTIONS]           start the local web UI (http://127.0.0.1:8787)
    risa --check [OPTIONS]        verify Ollama connectivity + model availability
    risa --list-models [OPTIONS]  list models installed on the Ollama server

Modes: ``--mode plan|build|auto``. ``--safe`` keeps the legacy approval overlay
handle. ``--ui`` shows the full staged startup sequence, then serves the UI.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import MODES, AgentConfig, load_config
from .loop import run_agent
from .ollama import OllamaClient, OllamaError
from .web import serve
from .workspace import Workspace, WorkspaceError

UI_DEFAULT_HOST = "127.0.0.1"
UI_DEFAULT_PORT = 8787


def _print_step(msg: str) -> None:
    print(msg, flush=True)


def _print_error(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="risa",
        description="A.S.C.S. - A Smart Coding System (local Ollama backend).",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Development task. If omitted, you are prompted for one.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=[m.lower() for m in MODES],
        help="Agent mode: plan (inspect+plan only), build (plan then implement), "
        f"or auto (fully autonomous). Default: AUTO.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Repository/workspace to operate on (default: current directory).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model to use (default: qwen2.5-coder:14b).",
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
        help="Require approval before modifications and command execution "
        "(legacy SAFE overlay).",
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
        "--request-timeout",
        type=int,
        default=None,
        help="Per-model-request timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--keep-alive",
        default=None,
        help="Ollama keep_alive for the model between steps, e.g. '30m'.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Start the local web UI instead of a one-shot session.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"Web UI bind host (default: {UI_DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Web UI bind port (default: {UI_DEFAULT_PORT}).",
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


def build_config(args: argparse.Namespace) -> "AgentConfig":
    """Resolve the effective config from CLI args (raises ValueError)."""
    if args.safe and args.auto:
        raise ValueError("Cannot use --safe and --auto together.")
    overrides: dict = {
        "workspace": args.workspace or ".",
    }
    for key, value in (
        ("model", args.model),
        ("ollama_base_url", args.base_url),
        ("max_iterations", args.max_iterations),
        ("command_timeout", args.command_timeout),
        ("request_timeout", args.request_timeout),
        ("keep_alive", args.keep_alive),
        ("ui_host", args.host),
        ("ui_port", args.port),
    ):
        if value is not None:
            overrides[key] = value
    if args.verbose:
        overrides["verbose"] = True
    if args.mode:
        overrides["mode"] = args.mode.upper()
    if args.safe:
        overrides["mode"] = "SAFE"
    if args.auto:
        overrides["mode"] = "AUTO"
    return load_config(**overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = build_config(args)
    except (ValueError, WorkspaceError) as exc:
        _print_error(str(exc))
        return 2

    client = OllamaClient(
        base_url=config.ollama_base_url,
        model=config.model,
        request_timeout=config.request_timeout,
        keep_alive=config.keep_alive,
    )

    # -- standalone preflight commands -------------------------------------
    if args.list_models:
        return _cmd_list_models(client)
    if args.check:
        return _cmd_check(client)

    # -- web UI mode --------------------------------------------------------
    if args.ui:
        return _cmd_ui(config, client)

    # -- preflight connectivity --------------------------------------------
    from .boot import boot

    report = boot(
        workspace_path=str(config.workspace),
        prewarm=False,
        **{
            f: getattr(config, f)
            for f in config.__dataclass_fields__
            if f not in ("workspace", "prewarm")
        },
    )
    if not report.ok:
        _print_error(report.error)
        return 1

    _print_step(f"risa v{__version__} — Ollama model: {config.model}")
    _print_step(f"Workspace: {config.workspace}")
    _print_step(f"Mode: {config.mode}")

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
        _print_step("\nCancelled.")
        return 130
    except OllamaError as exc:
        _print_step("\nSTATUS: FAILED")
        _print_error(str(exc))
        return 1

    print(f"\nSTATUS: {result.status.upper()}")
    if result.plan is not None:
        print("Plan:")
        print(result.plan.to_text())
    if result.iterations:
        print(f"Iterations: {result.iterations}")
    if result.summary:
        print(f"Summary: {result.summary}")
    if result.error:
        _print_error(result.error)

    return 0 if result.is_complete else 1


def _cmd_ui(config, client) -> int:
    """Boot sequence then serve the local web UI."""
    from .boot import boot, print_boot

    report = boot(
        workspace_path=str(config.workspace),
        prewarm=config.prewarm,
        **{
            f: getattr(config, f)
            for f in config.__dataclass_fields__
            if f not in ("workspace", "prewarm")
        },
    )
    print_boot(report)
    if not report.ok:
        return 1
    try:
        workspace = Workspace(config.workspace)
    except WorkspaceError as exc:
        _print_error(str(exc))
        return 2
    try:
        serve(config, client, workspace, block=True)
    except KeyboardInterrupt:
        pass
    return 0


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