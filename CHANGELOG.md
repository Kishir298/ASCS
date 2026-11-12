# Changelog

## [0.3.0] — 2026-11-11

### Added

- `ascs setup` and `ascs doctor` subcommands (previously only `--doctor` and
  `--check` flags on the `risa` parser).
- `ascs --help` now presents the installed program name; `risa` remains a
  compatibility alias.
- Cross-platform release pipeline with Windows x64/ARM64, macOS Intel/ARM64 and
  Linux x64/ARM64 executables, Python wheels, checksums and CycloneDX SBOM.
- Machine-readable `compatibility.json`.
- Platform smoke CI job that runs the installed commands from a different
  directory on Ubuntu, Windows and macOS.
- Documentation: installation, building, releases, security, compatibility.

### Fixed

- `--help` usage line advertised `risa` even when invoked as `ascs`.
- Cross-platform runtime: POSIX `run_command` now executes through the shell
  (`/bin/sh`, or `$SHELL`) so compound commands, pipes and redirection behave
  as they do through cmd.exe on Windows. Previously POSIX used `shlex.split`
  without a shell, silently breaking `a && b`.
- `ascs doctor` reports the platform as supported on macOS/Linux instead of
  warning that the runtime is Windows-only; the capability was already present.

### Known limitations

- Windows ARM64 is experimental.
- Standalone binaries are unsigned unless signing secrets are configured.
