"""Frozen-executable entry point for A.S.C.S. (used only by PyInstaller)."""

from agent.launch import main

if __name__ == "__main__":
    raise SystemExit(main())
