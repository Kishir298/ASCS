"""Workspace model with strict containment enforcement.

Every agent session has an explicit workspace root. All file operations are
resolved against this root and rejected if they would escape it via absolute
paths, ``..`` segments, or symlinks/junctions.
"""

from __future__ import annotations

import os
from pathlib import Path

# Directory names never surfaced by list/search tools (generated artifacts,
# VCS metadata, virtual environments).
IGNORED_NAMES = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".hypothesis",
    "dist",
    "build",
    ".tox",
    ".nox",
    ".eggs",
    ".ipynb_checkpoints",
    ".idea",
    ".vscode",
}

IGNORED_SUFFIXES = (".egg-info",)


class WorkspaceError(Exception):
    """Raised for any workspace boundary violation or bad path."""


def should_ignore(name: str) -> bool:
    """True if a directory/file entry should be hidden from the model."""
    if name in IGNORED_NAMES:
        return True
    return name.lower().endswith(IGNORED_SUFFIXES)


class Workspace:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        try:
            self._root = Path(root).expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise WorkspaceError(f"Workspace root is not accessible: {root!r}") from exc
        if not self._root.is_dir():
            raise WorkspaceError(f"Workspace root is not a directory: {self._root}")

    @property
    def root(self) -> Path:
        return self._root

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Workspace({self._root})"

    def resolve(self, path: str | os.PathLike[str] | None) -> Path:
        """Resolve a user/model-supplied path against the workspace root.

        Safe for paths that do not exist yet. Raises ``WorkspaceError`` on any
        escape attempt (absolute outside, ``..`` traversal, symlink/junction
        pointing outside the root).
        """
        if path is None or str(path).strip() == "":
            path = "."
        raw = os.path.expanduser(str(path))
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        candidate = Path(os.path.normpath(candidate))

        real_root = os.path.realpath(str(self._root))
        real_candidate = os.path.realpath(str(candidate))
        try:
            common = os.path.commonpath(
                [os.path.normcase(real_root), os.path.normcase(real_candidate)]
            )
        except ValueError as exc:
            raise WorkspaceError(
                f"Path {path!r} resolves outside the workspace: {candidate}"
            ) from exc
        if common != os.path.normcase(real_root):
            raise WorkspaceError(
                f"Path {path!r} resolves outside the workspace root "
                f"{self._root} (resolved to {candidate})"
            )
        return Path(real_candidate)

    def is_inside(self, path: str | os.PathLike[str]) -> bool:
        try:
            self.resolve(path)
        except WorkspaceError:
            return False
        return True

    # -- file/listing helpers ----------------------------------------------

    @staticmethod
    def iter_files(root: Path) -> "list[Path]":
        """Yield regular files under ``root``, skipping ignored directories."""
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not should_ignore(d)]
            for name in filenames:
                if should_ignore(name):
                    continue
                files.append(Path(dirpath) / name)
        return files