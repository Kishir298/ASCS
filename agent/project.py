"""A.S.C.S. project intelligence.

Turns a repository from a collection of files into a persistent, machine-
readable description of the project itself:

* :class:`ProjectManifest` - the project's identity, languages, frameworks,
  package managers, dependencies, entry points, tests, config and docs.
* :func:`scan` - discover that information from a workspace without modifying
  anything.
* :class:`ProjectStore` - a persistent, versioned container that holds the
  manifest, the :class:`~agent.context.ProjectIndex` and per-project task state
  (used by the task engine in later phases). It survives process restarts and
  is updated incrementally.

Design rules (from the master plan):

* Read-only: scanning never modifies project files.
* Ignore noise: ``.git``, ``.venv``, build artifacts, caches, binaries.
* Persist what matters, retrieve only what the task needs.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .context import (DEFAULT_IGNORED_DIRS, DEFAULT_INDEX_FILE, DEFAULT_STATE_DIR,
                      ContextError, ProjectIndex)
from .tasks import TaskGraph, TaskGraphError

MANIFEST_FILE = "project_manifest.json"
TASK_STATE_FILE = "task_state.json"
MANIFEST_VERSION = 1

# Extension -> language (single source of truth for scanning).
_LANGUAGE_BY_EXT = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sh": "shell",
    ".bash": "shell",
    ".ps1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".md": "markdown",
    ".rst": "rST",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
}

# Package manager manifests and the language they imply.
_PACKAGE_MANIFESTS = {
    "pyproject.toml": "python",
    "requirements.txt": "python",
    "setup.py": "python",
    "setup.cfg": "python",
    "Pipfile": "python",
    "poetry.lock": "python",
    "package.json": "javascript",
    "yarn.lock": "javascript",
    "package-lock.json": "javascript",
    "pnpm-lock.yaml": "javascript",
    "go.mod": "go",
    "Cargo.toml": "rust",
    "Gemfile": "ruby",
    "composer.json": "php",
    "pom.xml": "java",
    "build.gradle": "java",
    "Gradle.xml": "java",
}

# Common config file names that matter for understanding a project.
_CONFIG_FILES = {
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "pytest.ini",
    "requirements.txt",
    "requirements-dev.txt",
    "Pipfile",
    ".flake8",
    ".editorconfig",
    ".env.example",
    "package.json",
    "tsconfig.json",
    ".eslintrc",
    ".eslintrc.json",
    ".prettierrc",
    "jest.config.js",
    "jest.config.ts",
    "vite.config.js",
    "vite.config.ts",
    "webpack.config.js",
    "go.mod",
    "Cargo.toml",
    "Makefile",
    "Dockerfile",
}

# Documentation files that are cheap to include in the manifest.
_DOC_FILES = {
    "README",
    "README.md",
    "README.rst",
    "CONTRIBUTING",
    "CONTRIBUTING.md",
    "CHANGELOG",
    "CHANGELOG.md",
    "CHANGELOG.rst",
    "LICENSE",
    "NOTICE",
}


@dataclass
class ProjectManifest:
    """Machine-readable description of a project.

    The manifest is the persistent, reusable "project map" from the master
    plan: identity, languages, frameworks, dependencies, entry points, tests,
    directories, configuration, git state and indexing state.
    """

    version: int = MANIFEST_VERSION
    root: str = ""
    name: str = ""
    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)
    important_dirs: list[str] = field(default_factory=list)
    git: dict = field(default_factory=dict)  # {"repository": bool, "branch": str}
    indexed: bool = False
    index_version: int = 0
    toolchain: str = ""  # rendered detect_toolchain() summary
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ProjectManifest":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class ScanResult:
    """Raw discovery result from a single scan pass."""

    languages: set[str] = field(default_factory=set)
    frameworks: set[str] = field(default_factory=set)
    package_managers: set[str] = field(default_factory=set)
    dependencies: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)
    important_dirs: list[str] = field(default_factory=list)


class ProjectScanner:
    """Read-only discovery of the project's identity and structure."""

    def __init__(
        self,
        root: str | Path,
        *,
        ignored_dirs: Iterable[str] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ContextError(f"Project root is not a directory: {self.root}")
        self.ignored = set(DEFAULT_IGNORED_DIRS)
        if ignored_dirs:
            self.ignored.update(ignored_dirs)
        self.ignored.add(".ascs")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def scan(self) -> ScanResult:
        """Walk the project once and collect its high-level shape."""
        result = ScanResult()

        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root)
            if any(part in self.ignored for part in relative.parts):
                continue

            name = path.name
            suffix = path.suffix.lower()

            lang = _LANGUAGE_BY_EXT.get(suffix)
            if lang:
                result.languages.add(lang)

            if name in _PACKAGE_MANIFESTS:
                result.package_managers.add(name)
                result.languages.add(_PACKAGE_MANIFESTS[name])
                result.dependencies.extend(self._parse_dependencies(path))

            if name in _CONFIG_FILES:
                result.config_files.append(relative.as_posix())

            if name in _DOC_FILES:
                result.docs.append(relative.as_posix())

            if self._looks_like_test(path):
                result.tests.append(relative.as_posix())

            if self._looks_like_entry(path):
                result.entry_points.append(relative.as_posix())

        # Frameworks from manifest content.
        self._detect_frameworks(result)

        # Important top-level directories (source, tests, tooling).
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            name = child.name.lower()
            if name in {"src", "lib", "app", "tests", "test", "spec", "docs", "bin", "scripts", "tools"}:
                result.important_dirs.append(child.name)

        result.languages = self._rank_languages(result.languages)
        result.frameworks = sorted(result.frameworks)
        result.dependencies = sorted(set(result.dependencies))
        result.tests = sorted(result.tests)
        result.entry_points = sorted(result.entry_points)
        result.config_files = sorted(result.config_files)
        result.docs = sorted(result.docs)
        result.important_dirs = sorted(result.important_dirs)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_dependencies(self, manifest: Path) -> list[str]:
        """Extract a coarse dependency list from a package manifest."""
        name = manifest.name
        try:
            text = manifest.read_text(encoding="utf-8")
        except OSError:
            return []

        deps: list[str] = []

        if name == "requirements.txt":
            for raw in text.splitlines():
                line = raw.strip()
                if not line or line.startswith(("#", "-")):
                    continue
                deps.append(line.split("==")[0].split("<")[0].split(">")[0].split("[")[0].strip())
            return deps

        if name == "pyproject.toml":
            try:
                import tomllib

                data = tomllib.loads(text)
            except (ValueError, OSError):
                data = {}
            for key in ("dependencies", "dev-dependencies"):
                entries = data.get("project", {}).get(key) or data.get(key) or []
                for entry in entries if isinstance(entries, list) else []:
                    if isinstance(entry, str):
                        normalized = entry.split(";")[0].strip()
                        pkg = normalized.split("(")[0].strip("'\" ")
                        deps.append(pkg)
            return [d for d in deps if d]

        if name == "package.json":
            import re as _re

            m = _re.search(r'"dependencies"\s*:\s*\{([^}]*)\}', text)
            if m:
                for pair in m.group(1).split(","):
                    k = pair.split(":")[0].strip().strip('"')
                    if k:
                        deps.append(k)
            m = _re.search(r'"devDependencies"\s*:\s*\{([^}]*)\}', text)
            if m:
                for pair in m.group(1).split(","):
                    k = pair.split(":")[0].strip().strip('"')
                    if k:
                        deps.append(k)
            return deps

        if name in ("Cargo.toml", "go.mod"):
            for raw in text.splitlines():
                line = raw.strip()
                if line and not line.startswith(("[", "#", "module", "go ")):
                    deps.append(line.split(" ")[0].split("=")[0].strip())
            return deps

        return deps

    def _looks_like_test(self, path: Path) -> bool:
        name = path.name.lower()
        if name.startswith("test_") or name.endswith("_test.py") or name.endswith(".spec.js"):
            return True
        return name.endswith("_test.go") or name.endswith("_test.py")

    def _looks_like_entry(self, path: Path) -> bool:
        name = path.name
        if name in {"main.py", "__main__.py", "app.py", "cli.py", "manage.py",
                    "index.js", "index.ts", "main.go", "main.rs"}:
            return True
        return False

    def _detect_frameworks(self, result: ScanResult) -> None:
        lowered = [d.lower() for d in result.dependencies] + [
            c.lower() for c in result.config_files
        ]
        registry = {
            "django": "Django",
            "flask": "Flask",
            "fastapi": "FastAPI",
            "pytest": "pytest",
            "react": "React",
            "vue": "Vue",
            "svelte": "Svelte",
            "angular": "Angular",
            "express": "Express",
            "next": "Next.js",
            "tensorflow": "TensorFlow",
            "torch": "PyTorch",
            "requests": "requests",
            "click": "Click",
        }
        for key, fmt in registry.items():
            if any(key in item for item in lowered):
                result.frameworks.add(fmt)

    @staticmethod
    def _rank_languages(languages: set[str]) -> list[str]:
        """Order languages by specificity; deterministic."""
        names = sorted(languages, key=lambda n: (-len(n), n))
        return names or ["unknown"]


def scan(
    root: str | Path,
    *,
    ignored_dirs: Iterable[str] | None = None,
) -> ProjectManifest:
    """Scan ``root`` and return a ready-to-persist :class:`ProjectManifest`."""
    scanner = ProjectScanner(root, ignored_dirs=ignored_dirs)
    result = scanner.scan()
    name = Path(root).resolve().name or "project"
    git = _git_info(Path(root).resolve())
    from .toolchain import detect_toolchain, toolchain_to_text

    toolchain = detect_toolchain(Path(root).resolve())
    return ProjectManifest(
        root=str(Path(root).resolve()),
        name=name,
        languages=result.languages,
        frameworks=result.frameworks,
        package_managers=sorted(result.package_managers),
        dependencies=result.dependencies,
        entry_points=result.entry_points,
        tests=result.tests,
        config_files=result.config_files,
        docs=result.docs,
        important_dirs=result.important_dirs,
        git=git,
        indexed=False,
        toolchain=toolchain_to_text(toolchain),
        updated_at=time.time(),
    )


def _git_info(root: Path) -> dict:
    import subprocess

    if not (root / ".git").exists():
        return {"repository": False, "branch": ""}
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return {"repository": True, "branch": (branch.stdout.strip() or "unknown")}
    except (OSError, subprocess.SubprocessError):
        return {"repository": True, "branch": "unknown"}


class ProjectStore:
    """Persistent, versioned project data that survives process restarts.

    Holds:

    * the :class:`ProjectManifest` (``project_manifest.json``),
    * the :class:`~agent.context.ProjectIndex` (``context_index.json``),
    * per-project task state (``task_state.json``, used by the task engine).

    The store is created once per project (in ``.ascs``) and updated
    incrementally: the manifest is rescanned only when something relevant to
    discovery changed, and the file index reuses unchanged records.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        state_dir: str = DEFAULT_STATE_DIR,
        ignored_dirs: Iterable[str] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ContextError(f"Project root is not a directory: {self.root}")

        self.state_dir = self.root / state_dir
        self.manifest_path = self.state_dir / MANIFEST_FILE
        self.task_state_path = self.state_dir / TASK_STATE_FILE
        self.index = ProjectIndex(
            self.root, state_dir=state_dir, ignored_dirs=ignored_dirs
        )
        self.ignored_dirs = ignored_dirs

    # -- manifest ---------------------------------------------------------

    def load_manifest(self) -> ProjectManifest | None:
        if not self.manifest_path.exists():
            return None
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        manifest = ProjectManifest.from_dict(payload)
        if manifest.root and Path(manifest.root) != self.root:
            return None  # moved project; do not trust a stale manifest
        return manifest

    def refresh(
        self,
        *,
        force: bool = False,
        rescan_threshold_s: float = 300.0,
    ) -> ProjectManifest:
        """Ensure the manifest and file index are current.

        * If no manifest exists, or it is older than ``rescan_threshold_s``,
          a fresh scan runs.
        * The file index is updated incrementally (``ProjectIndex.update``).
        """
        manifest = self.load_manifest()
        now = time.time()
        stale = manifest is None or force or (now - (manifest.updated_at or 0)) > rescan_threshold_s

        if stale:
            manifest = scan(self.root, ignored_dirs=self.ignored_dirs)
            if manifest is not None:
                self.save_manifest(manifest)

        self.index.update()
        if manifest is not None and not manifest.indexed:
            manifest.indexed = True
            manifest.index_version = 1
            self.save_manifest(manifest)

        return manifest if manifest is not None else self.load_manifest()

    def save_manifest(self, manifest: ProjectManifest) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self.manifest_path, manifest.to_dict())

    def snapshot(self) -> dict:
        """A compact, serializable snapshot used by the web/CLI and integrations."""
        manifest = self.load_manifest()
        return {
            "root": str(self.root),
            "name": manifest.name if manifest else self.root.name,
            "languages": manifest.languages if manifest else [],
            "frameworks": manifest.frameworks if manifest else [],
            "files_indexed": len(self.index.records),
            "indexed": bool(manifest and manifest.indexed),
            "manifest_present": manifest is not None,
            "has_git": bool(manifest and manifest.git.get("repository")),
        }

    # -- task state (Phase 3 uses this) -----------------------------------

    def save_task_graph(self, graph: TaskGraph) -> None:
        """Persist a task graph to ``task_state.json`` atomically."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self.task_state_path, graph.to_dict())

    def load_task_graph(self) -> TaskGraph | None:
        """Load the persisted task graph, or ``None`` if absent/corrupt."""
        if not self.task_state_path.exists():
            return None
        try:
            payload = json.loads(self.task_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return TaskGraph.from_dict(payload)
        except (TaskGraphError, TypeError, ValueError):
            return None

    def _atomic_write(self, path: Path, payload) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)


def open_project(
    root: str | Path,
    *,
    ignored_dirs: Iterable[str] | None = None,
) -> ProjectStore:
    """Open (and lazily refresh) the persistent project store for ``root``."""
    return ProjectStore(root, ignored_dirs=ignored_dirs)


def project_prompt_text(store: ProjectStore, *, max_lines: int = 25) -> str:
    """Render a compact project-intelligence block for the system prompt.

    Derived entirely from the persisted manifest and file index, so repeated
    sessions do not rescan the repository from scratch.
    """
    manifest = store.load_manifest()
    lines: list[str] = []
    if manifest is None:
        return "- Project not yet scanned; inspect before editing."

    lines.append(f"- Project: {manifest.name}")
    if manifest.languages:
        lines.append(f"- Languages: {', '.join(manifest.languages[:8])}")
    if manifest.toolchain:
        lines.append(f"- Toolchain: {manifest.toolchain}")
    if manifest.frameworks:
        lines.append(f"- Frameworks: {', '.join(manifest.frameworks[:8])}")
    if manifest.package_managers:
        lines.append(f"- Package managers: {', '.join(manifest.package_managers[:6])}")
    if manifest.entry_points:
        joined = ", ".join(f.split("/")[-1] for f in manifest.entry_points[:6])
        lines.append(f"- Entry points: {joined}")
    if manifest.tests:
        tests = ", ".join(f.replace("-", "") for f in manifest.tests[:6])
        lines.append(f"- Tests: {tests}")
    indexed = len(store.index.records)
    lines.append(f"- Files indexed: {indexed}")

    # Related tests/deps hint via a couple of glanceable summaries.
    if indexed:
        summaries = store.index.file_summaries(limit=8)
        if summaries:
            lines.append("- Key files:")
            for summary in summaries:
                lines.append(f"    {summary}")

    return "\n".join(lines[:max_lines])


__all__ = [
    "MANIFEST_FILE",
    "MANIFEST_VERSION",
    "ProjectManifest",
    "ProjectScanner",
    "ProjectStore",
    "ScanResult",
    "open_project",
    "project_prompt_text",
    "scan",
]