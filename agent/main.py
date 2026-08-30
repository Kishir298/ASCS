"""Command-line entry point for A.S.C.S. (``risa``).

Subcommands/behaviors:
    risa [OPTIONS] [TASK]         run a one-shot agent session
    risa --ui [OPTIONS]           start the local web UI (http://127.0.0.1:8787)
    risa --check [OPTIONS]        verify Ollama connectivity + model availability
    risa --doctor [OPTIONS]       run the full read-only diagnostic suite
    risa --list-models [OPTIONS]  list models installed on the Ollama server

Modes: ``--mode plan|build|auto``. ``--safe`` keeps the legacy approval overlay
handle. ``--ui`` shows the full staged startup sequence, then serves the UI.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import DEFAULT_UI_HOST, DEFAULT_UI_PORT, MODES, AgentConfig, load_config
from .loop import run_agent, run_graph_agent
from .ollama import OllamaClient, OllamaError
from .web import serve
from .workspace import Workspace, WorkspaceError

UI_DEFAULT_HOST = DEFAULT_UI_HOST
UI_DEFAULT_PORT = DEFAULT_UI_PORT


def _print_step(msg: str) -> None:
    print(msg, flush=True)


def _print_error(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)


def _boot_progress(phase: str, message: str, elapsed: float) -> None:
    """Live-print each finished startup stage with its real timing."""
    stamp = f"[{elapsed:6.2f}s]" if elapsed > 0 else "[     ]"
    print(f"{stamp} {message}", flush=True)


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
        help="Ollama model to use (default: qwen3:14b).",
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
        "--tasks",
        action="store_true",
        help="Use the task-graph engine: plan the objective into a DAG of "
        "tasks and execute/verify them one by one (resumable via .ascs).",
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
        "--num-ctx",
        type=int,
        default=None,
        help="Qwen3 context window size in tokens (default: 32768).",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="Max tokens generated per request (default: 8192).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Retries after the initial Ollama attempt for transient "
        "failures (default: 2, i.e. 3 total attempts).",
    )
    parser.add_argument(
        "--backoff-s",
        type=float,
        default=None,
        help="Base backoff seconds between Ollama request retries (default: 2.0).",
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
        "--doctor",
        action="store_true",
        help="Run the full read-only diagnostic suite (environment, config, "
        "workspace, Ollama, model, tools, context, git, tests), then exit.",
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
        ("num_ctx", args.num_ctx),
        ("num_predict", args.num_predict),
        ("max_retries", args.max_retries),
        ("backoff_s", args.backoff_s),
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
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
    )

    # -- standalone preflight commands -------------------------------------
    if args.list_models:
        return _cmd_list_models(client)
    if args.check:
        return _cmd_check(client)
    if args.doctor:
        return _cmd_doctor(config)

    # -- web UI mode --------------------------------------------------------
    if args.ui:
        return _cmd_ui(config, client)

    # -- preflight connectivity --------------------------------------------
    from .boot import boot

    _print_step("A.S.C.S. starting...")
    report = boot(
        workspace_path=str(config.workspace),
        prewarm=False,
        progress=_boot_progress,
        **{
            f: getattr(config, f)
            for f in config.__dataclass_fields__
            if f not in ("workspace", "prewarm")
        },
    )
    if not report.ok:
        _print_error(report.error)
        return 1
    _print_step("A.S.C.S. ready.")

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
        if args.tasks:
            result = run_graph_agent(config, client, task, log=_print_step)
        else:
            result = run_agent(config, client, task, log=_print_step)
    except KeyboardInterrupt:
        _print_step("\nCancelled.")
        return 130
    except OllamaError as exc:
        _print_step("\nSTATUS: FAILED")
        _print_error(str(exc))
        return 1

    print(f"\nSTATUS: {result.status.upper()}")
    plan = getattr(result, "plan", None)
    if plan:
        print("Plan:")
        print(plan)
    if getattr(result, "report", ""):
        print("\nReport:")
        print(result.report)
    if getattr(result, "iterations", None):
        print(f"Iterations: {result.iterations}")
    elif isinstance(getattr(result, "task_count", None), int) and result.task_count:
        print(f"Tasks: {result.task_count}")
    if getattr(result, "summary", ""):
        print(f"Summary: {result.summary}")
    if getattr(result, "error", ""):
        _print_error(result.error)

    return 0 if result.is_complete else 1


def _cmd_ui(config, client) -> int:
    """Boot sequence then serve the local web UI."""
    from .boot import boot, boot_error_message

    _print_step("A.S.C.S. starting...")
    report = boot(
        workspace_path=str(config.workspace),
        prewarm=config.prewarm,
        progress=_boot_progress,
        **{
            f: getattr(config, f)
            for f in config.__dataclass_fields__
            if f not in ("workspace", "prewarm")
        },
    )
    if report.ok:
        _print_step("A.S.C.S. ready.")
    else:
        _print_step("A.S.C.S. startup FAILED.")
        _print_error(boot_error_message(report))
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
    except OllamaError as exc:
        _print_error(f"Could not connect to Ollama at {client.base_url}.\n  {exc}")
        _print_error("Make sure Ollama is running.")
        return 1
    except Exception as exc:
        _print_error(f"Could not reach Ollama at {client.base_url}: {exc}")
        return 1
    print("Installed models:")
    for name in models:
        print(f"  {name}")
    return 0


def _cmd_check(client: OllamaClient) -> int:
    try:
        client.check_connectivity(timeout=5)
    except OllamaError as exc:
        _print_error(f"Could not connect to Ollama at {client.base_url}.\n  {exc}")
        _print_error("Make sure Ollama is running.")
        return 1
    except Exception as exc:
        _print_error(f"Could not reach Ollama at {client.base_url}: {exc}")
        return 1
    try:
        models = client.list_models(timeout=10)
    except Exception as exc:
        _print_error(f"Could not list models at {client.base_url}: {exc}")
        return 1
    print("Ollama: OK")
    print(f"Model {client.model!r}: "
          f"{'available' if client.model in models else 'NOT INSTALLED'}")
    return 0


def _cmd_doctor(config) -> int:
    from .doctor import doctor, print_doctor

    report = doctor(workspace=config.workspace)
    print_doctor(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())