# Building ASCS

## Prerequisites

- Python 3.12+
- `python -m pip install build pyinstaller`
- Windows: `windows-curses>=2.3` for TUI builds.

## Wheel and sdist

```bash
python -m build --wheel --sdist
```

## Standalone executable

```bash
python -m pip install -e . pyinstaller
python -m PyInstaller --onefile --name ascs \
  --collect-submodules agent --collect-data agent \
  --distpath dist scripts/standalone_entry.py
```

Build per target OS/architecture; no cross-compilation.

## Tests and lint

```bash
python -m pytest -o addopts= -q
ruff check --select F agent tests
```

## Release automation

`.github/workflows/release.yml` validates the tag/version/manifest, runs tests,
builds wheels and executables for Windows x64/ARM64, macOS Intel/ARM64 and
Linux x64/ARM64, writes SHA-256 checksums and a CycloneDX SBOM, and publishes
stable/beta releases. Manual dispatch only uploads artifacts.
