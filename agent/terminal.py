"""Terminal-native entry point for A.S.C.S.

A.S.C.S. runs as a terminal application. The historical ``--ui`` option is
retained as a compatibility alias for the terminal TUI and no longer starts
the browser-based UI.
"""

from __future__ import annotations

import sys

from .config import load_config
from .main import (
    _cmd_check,
    _cmd_doctor,
    _cmd_list_models,
    _cmd_tui,
    build_parser,
)
from .ollama import OllamaClient


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

    if not has_diagnostic and not has_tui:
        normalized.append("--tui")

    elif has_tui and "--tui" not in normalized:
        normalized.append("--tui")

    return normalized


def main(argv: list[str] | None = None) -> int:
    """Start A.S.C.S. as a terminal-native application."""
    normalized = normalize_argv(argv)
    args = build_parser().parse_args(normalized)

    try:
        # Use the existing configuration system. Do not duplicate config
        # construction here.
        from .main import build_config

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

    print(ASCII_BANNER, flush=True)

    return _cmd_tui(config, client)


if __name__ == "__main__":
    raise SystemExit(main())