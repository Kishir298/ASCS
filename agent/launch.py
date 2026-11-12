"""Installed command, preserving the legacy terminal entry point."""
def main():
    from agent.terminal.entry import main as terminal_main
    return terminal_main()
