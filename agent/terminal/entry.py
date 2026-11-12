"""Terminal-native entry point for A.S.C.S.

A.S.C.S. runs as a terminal application. The historical ``--ui`` option is
retained as a compatibility alias for the terminal TUI and no longer starts
the browser-based UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent.main import (
    _cmd_check,
    _cmd_doctor,
    _cmd_list_models,
    _cmd_tui,
    build_parser,
)
from agent.models.client import OllamaClient


def _setup() -> int:
    """Create the per-user config directory and persist safe defaults.

    Offline only: no model download, no network probe, existing state preserved.
    """
    from agent.config import (
        DEFAULT_MODEL,
        DEFAULT_PROVIDER,
        load_tui_state,
        save_tui_state,
        tui_state_path,
    )

    path = tui_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    existing = load_tui_state(path)
    defaults = {"provider": DEFAULT_PROVIDER, "model": DEFAULT_MODEL, "mode": "AUTO"}
    merged = {**defaults, **existing}
    save_tui_state(merged, path)
    print("A.S.C.S. setup complete.")
    print("Configuration:", path)
    print("Provider:", merged.get("provider"), "Model:", merged.get("model"))
    print("Next: ascs doctor, then ascs to launch the terminal interface.")
    return 0


def _doctor(workspace: str | None = None) -> int:
    from agent.main import build_config

    args = build_parser().parse_args(["--doctor"])
    if workspace:
        args.workspace = workspace
    return _cmd_doctor(build_config(args))

ASCII_BANNER = r"""
    █████╗ ███████╗ ██████╗███████╗
   ██╔══██╗██╔════╝██╔════╝██╔════╝
   ███████║███████╗██║     ███████╗
   ██╔══██║╚════██║██║     ╚════██║
   ██║  ██║███████║╚██████╗███████║
   ╚═╝  ╚═╝╚══════╝ ╚═════╝╚══════╝

        A  S M A R T  C O D I N G  S Y S T E M
""".strip("\n")


def normalize_argv(argv: list[str] | None) -> list[str]:
    """Make the terminal TUI the default application mode.

    ``--ui`` is treated as a backwards-compatible alias for ``--tui``.
    Diagnostic commands remain one-shot commands.
    """
    args = list(sys.argv[1:] if argv is None else argv)

    normalized: list[str] = []
    has_tui = False
    has_ui = False
    has_diagnostic = False

    for arg in args:
        if arg == "--ui":
            has_ui = True
            continue

        if arg == "--tui":
            has_tui = True

        if arg in {"--check", "--doctor", "--list-models"}:
            has_diagnostic = True

        normalized.append(arg)

    if has_ui:
        has_tui = True

    if not has_diagnostic and not has_tui or has_tui and "--tui" not in normalized:
        normalized.append("--tui")

    return normalized


def main(argv: list[str] | None = None) -> int:
    """Start A.S.C.S. as a terminal-native application."""
    from agent import pi
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "setup":
        return _setup()
    if raw and raw[0] == "doctor":
        return _doctor(raw[1] if len(raw) > 1 and not raw[1].startswith("-") else None)
    if pi.enabled() or "--pi" in raw or "--pi-status" in raw:
        from agent.main import main as task_main
        return task_main(raw)
    normalized = normalize_argv(argv)
    parser = build_parser()
    invoked = Path(sys.argv[0]).stem.lower()
    parser.prog = invoked if invoked in {"ascs", "risa"} else "ascs"
    args = parser.parse_args(normalized)

    try:
        # Use the existing configuration system. Do not duplicate config
        # construction here.
        from agent.main import build_config

        config = build_config(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2

    # Diagnostic commands should behave exactly like the normal CLI.
    client = OllamaClient(
        base_url=config.ollama_base_url,
        model=config.model,
        request_timeout=config.request_timeout,
        keep_alive=config.keep_alive,
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
    )

    if args.list_models:
        return _cmd_list_models(client)

    if args.check:
        return _cmd_check(client)

    if args.doctor:
        return _cmd_doctor(config)

    try:
        # Box-drawing banner needs UTF-8; legacy consoles (cp1252/cp437)
        # would crash here before the TUI even starts.
        print(ASCII_BANNER, flush=True)
    except UnicodeEncodeError:
        print("A.S.C.S. - A SMART CODING SYSTEM", flush=True)

    rc = _cmd_tui(config, client)
    from agent.models.proc import maybe_stop_ollama_on_exit

    maybe_stop_ollama_on_exit(config)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())