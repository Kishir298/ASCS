"""A.S.C.S. toolchain detection.

Detects the practical command/toolchain for a repository from evidence on
disk (manifest files, config files, directories) rather than hardcoded
assumptions, so ASCS can run the *right* test/build/lint/type-check commands
for a given language ecosystem instead of guessing.

``Toolchain`` is a small, immutable description of what commands likely apply
to this repository. Detection walks a small set of well-known marker files in
a deterministic order and returns one best-effort toolchain (plus flags).

Detection is read-only and cheap: it inspects only top-level marker files and
never scans the whole repository. It favours evidence over presence: for
example a ``package.json`` that contains a ``test`` script is preferred over a
bare ``package.json`` with no scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Marker file -> (language, package manager, [test commands], [lint commands])
# Order matters: the first matching marker wins. Rows later in the list are
# lower priority, so we only fall through to them if nothing higher matched.


@dataclass(frozen=True)
class Toolchain:
    """A detected repository toolchain (language + practical commands)."""

    language: str = "unknown"
    package_manager: str = ""
    test_commands: tuple[str, ...] = ()
    lint_commands: tuple[str, ...] = ()
    build_command: str = ""
    markers: tuple[str, ...] = ()

    @property
    def detected(self) -> bool:
        return self.language != "unknown"

    def __bool__(self) -> bool:
        return self.detected


def _has_script(root: Path, manifest: str, script: str) -> bool:
    """True when ``package.json`` has a runnable script entry for ``script``."""
    try:
        text = (root / manifest).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    import re

    m = re.search(r'"scripts"\s*:\s*\{([^}]*)\}', text)
    if not m:
        return False
    body = m.group(1)
    return re.search(rf'"{re.escape(script)}"\s*:', body) is not None


def _read_pyproject_tools(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract common tool commands from pyproject.toml, if present."""
    tests: list[str] = []
    lints: list[str] = []
    path = root / "pyproject.toml"
    if not path.exists():
        return tuple(tests), tuple(lints)
    try:
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError, ImportError):
        return tuple(tests), tuple(lints)

    def _present(table_key: str) -> bool:
        node = data
        for key in table_key.split("."):
            if not isinstance(node, dict) or key not in node:
                return False
            node = node[key]
        return bool(node)

    # pytest config -> "python -m pytest"
    if _present("tool.pytest"):
        tests.append("python -m pytest")
    if _present("tool.pytest.ini_options"):
        tests.append("python -m pytest")
    # ruff config -> "ruff check"
    if _present("tool.ruff"):
        lints.append("ruff check")
    if _present("tool.mypy") or _present("tool.mypy.ini_options"):
        lints.append("python -m mypy")
    return tuple(dict.fromkeys(tests)), tuple(dict.fromkeys(lints))


def detect_toolchain(root: str | Path) -> Toolchain:
    """Detect and return the best-effort toolchain for ``root``.

    Marker files are read from the repository root only. Detection is
    deterministic: the first recognised marker in priority order wins.
    """
    root = Path(root)
    tests: list[str] = []
    lints: list[str] = []
    marker: str | None = None

    # Priority 1: Python projects (pyproject / setup / requirements).
    if (root / "pyproject.toml").exists():
        py_tests, py_lints = _read_pyproject_tools(root)
        tests.extend(py_tests)
        lints.extend(py_lints)
        if not tests:
            # pytest absent from config; fall back to generic module run only
            # if a tests directory exists (else leave empty -> executor no-op).
            if (root / "tests").is_dir() or (root / "test").is_dir():
                tests.append("python -m pytest")
        if not lints and (root / ".flake8").exists():
            lints.append("flake8")
        marker = "pyproject.toml"
        return Toolchain(
            language="python",
            package_manager="pyproject.toml",
            test_commands=tuple(tests),
            lint_commands=tuple(lints),
            markers=(marker,),
        )

    if (root / "requirements.txt").exists() or (root / "setup.py").exists():
        marker = ("pyproject.toml" if (root / "pyproject.toml").exists()
                  else "requirements.txt" if (root / "requirements.txt").exists()
                  else "setup.py")
        if (root / "tests").is_dir() or (root / "test").is_dir():
            tests.append("python -m pytest")
        return Toolchain(
            language="python",
            package_manager=marker,
            test_commands=tuple(tests),
            lint_commands=tuple(lints),
            markers=(marker,),
        )

    # Priority 2: Node / TypeScript.
    if (root / "package.json").exists():
        marker = "package.json"
        # Use a defined "test" script when present; otherwise leave the test
        # command unset so the executor does not fabricate a passing run.
        if _has_script(root, "package.json", "test"):
            tests.append("npm test")
        if (root / "tsconfig.json").exists():
            lints.append("npx tsc --noEmit")
        elif _has_script(root, "package.json", "build"):
            lints.append("npm run build")
        return Toolchain(
            language="typescript" if (root / "tsconfig.json").exists() else "javascript",
            package_manager="package.json",
            test_commands=tuple(dict.fromkeys(tests)),
            lint_commands=tuple(dict.fromkeys(lints)),
            markers=(marker,),
        )

    # Priority 3: Rust.
    if (root / "Cargo.toml").exists():
        return Toolchain(
            language="rust",
            package_manager="Cargo.toml",
            test_commands=("cargo test",),
            lint_commands=("cargo clippy",),
            build_command="cargo build",
            markers=("Cargo.toml",),
        )

    # Priority 4: Go.
    if (root / "go.mod").exists():
        return Toolchain(
            language="go",
            package_manager="go.mod",
            test_commands=("go test ./...",),
            lint_commands=("go vet ./...",),
            build_command="go build ./...",
            markers=("go.mod",),
        )

    return Toolchain()


def toolchain_to_text(toolchain: Toolchain) -> str:
    """Render a compact human/model-readable description of the toolchain."""
    if not toolchain.detected:
        return "No standard toolchain detected; inspect before deciding on commands."
    parts = [f"Language: {toolchain.language}"]
    if toolchain.package_manager:
        parts.append(f"Package manager: {toolchain.package_manager}")
    if toolchain.test_commands:
        parts.append(f"Likely test commands: {'; '.join(toolchain.test_commands)}")
    if toolchain.lint_commands:
        parts.append(f"Likely lint/type commands: {'; '.join(toolchain.lint_commands)}")
    if toolchain.build_command:
        parts.append(f"Build: {toolchain.build_command}")
    return "; ".join(parts)


__all__ = [
    "Toolchain",
    "detect_toolchain",
    "toolchain_to_text",
]
