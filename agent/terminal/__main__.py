"""Run the terminal-native entry point: ``python -m agent.terminal``."""

from __future__ import annotations

from agent.terminal.entry import main

if __name__ == "__main__":
    raise SystemExit(main())
