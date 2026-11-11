# ASCS Installation

A.S.C.S. installs as `ascs` (terminal application) and the legacy `risa`
alias. It is standalone: CORE, RESCS and a Raspberry Pi are not required.
Python 3.12+.

## pip / pipx

```bash
python -m pip install .
python -m pip install ".[tui]"     # windows-curses on Windows
pipx install .                     # globally available command
```

## Windows

- Windows 10 22H2 and Windows 11 (x64; ARM64 where wheels exist).
- PowerShell 5.1 is supported; WSL and Bash are not required.
- `windows-curses` is installed automatically on Windows for the TUI.

## macOS

- Intel and Apple Silicon.
- If `ascs` is not on `PATH`, add the pipx bin directory (`~/.local/bin`) or
  your Python `bin` directory to `~/.zshrc`.

## Linux

- x86_64 and ARM64 where wheels exist.
- `~/.local/bin` must be on `PATH` for pipx/pip user installs.

## Commands

```bash
ascs --help
ascs setup        # private per-user state; offline, preserves existing values
ascs doctor       # read-only diagnostics
ascs              # interactive terminal interface
ascs --check      # Ollama connectivity/model check
risa --doctor     # legacy alias, same diagnostics
```

`ascs setup` never downloads a model and never contacts the network. State is
stored in the per-user config directory (`ascs doctor` prints the workspace
paths it uses).

## Models

ASCS supports Ollama locally and OpenAI-compatible cloud providers through
explicit configuration. Providers are never hard-coded; user-supplied API keys
are read from the environment and are not written into logs.

## CORE integration (optional)

CORE is optional. When enabled, device verification, authentication and CORE
permissions are enforced; ASCS remains fully usable when CORE is unreachable.

## Updating and uninstalling

Reinstall the package to update; configuration and state are preserved.
Uninstall with `pip uninstall coding-agent` or `pipx uninstall coding-agent`.
User data is not deleted automatically.

## Standalone executables

Release builds provide `ascs.exe` and POSIX `ascs` binaries. They include the
standard-library runtime; no Python installation is required.
