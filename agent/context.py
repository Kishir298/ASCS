"""
A.S.C.S. context engine.

Provides persistent project indexing, dependency-aware retrieval, and
deterministic context chunking for local LLM execution.

Design goals
------------
* Keep the complete project knowledge on disk rather than inside the model
  conversation.
* Retrieve only the information relevant to the current task.
* Keep every model request within a configurable token budget.
* Prefer semantic/code boundaries over arbitrary character slicing.
* Persist indexing metadata so later runs can resume quickly.
* Never modify project files while indexing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_CHUNK_TOKENS = 8192  # max-chunking for 300k: 2× prev, each shard still fits 65k ctx
DEFAULT_STATE_DIR = ".ascs"
DEFAULT_INDEX_FILE = "context_index.json"

# Files/directories that should not become model context.
DEFAULT_IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    "node_modules",
    "dist",
    "build",
    ".ascs",
    ".idea",
    ".vscode",
}

DEFAULT_IGNORED_FILES = {
    ".DS_Store",
}

TEXT_EXTENSIONS = {
    ".py",
    ".pyw",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".rst",
    ".ini",
    ".cfg",
    ".conf",
    ".html",
    ".css",
    ".scss",
    ".sql",
    ".sh",
    ".bash",
    ".ps1",
    ".bat",
    ".cmd",
    ".xml",
    ".csv",
    ".env.example",
}

# Conservative estimate. Ollama's tokenizer is model-specific, so ASCS uses
# this estimate for budgeting rather than pretending it has exact tokenizer
# parity with every installed model.
_TOKEN_CHARS = 4


@dataclass
class Symbol:
    """A source-code symbol discovered inside a file."""

    name: str
    kind: str
    line_start: int
    line_end: int


@dataclass
class FileRecord:
    """Persistent metadata for one indexed file."""

    path: str
    size: int
    modified_ns: int
    sha256: str
    language: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class ContextChunk:
    """A model-ready context unit."""

    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    estimated_tokens: int
    symbols: list[str] = field(default_factory=list)
    related_files: list[str] = field(default_factory=list)
    language: str = ""


@dataclass
class ContextBundle:
    """Complete context assembled for one model request."""

    task: str
    chunks: list[ContextChunk]
    estimated_tokens: int
    files: list[str]
    generated_at: float = field(default_factory=time.time)

    @property
    def text(self) -> str:
        sections: list[str] = []

        for chunk in self.chunks:
            sections.append(
                "\n".join(
                    [
                        f"===== {chunk.path}:{chunk.start_line}-{chunk.end_line} =====",
                        chunk.text,
                    ]
                )
            )

        return "\n\n".join(sections)


class ContextError(RuntimeError):
    """Raised when the context engine cannot perform an operation."""


class ProjectIndex:
    """
    Persistent project index.

    The index is intentionally stored inside ``.ascs`` so the model can retain
    knowledge across sessions without requiring the entire repository to be
    sent to Ollama every time.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        state_dir: str = DEFAULT_STATE_DIR,
        ignored_dirs: Iterable[str] | None = None,
        ignored_files: Iterable[str] | None = None,
    ) -> None:
        self.root = Path(root).resolve()

        if not self.root.exists():
            raise ContextError(f"Project root does not exist: {self.root}")

        if not self.root.is_dir():
            raise ContextError(f"Project root is not a directory: {self.root}")

        self.state_path = self.root / state_dir
        self.index_path = self.state_path / DEFAULT_INDEX_FILE

        self.ignored_dirs = set(DEFAULT_IGNORED_DIRS)
        if ignored_dirs:
            self.ignored_dirs.update(ignored_dirs)

        self.ignored_files = set(DEFAULT_IGNORED_FILES)
        if ignored_files:
            self.ignored_files.update(ignored_files)

        self.records: dict[str, FileRecord] = {}

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the persistent index if one exists."""

        if not self.index_path.exists():
            return

        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt/stale index must never prevent ASCS from starting.
            self.records = {}
            return

        records = payload.get("records", {})

        if not isinstance(records, dict):
            self.records = {}
            return

        loaded: dict[str, FileRecord] = {}

        for path, value in records.items():
            if not isinstance(value, dict):
                continue

            symbols = [
                Symbol(**symbol)
                for symbol in value.get("symbols", [])
                if isinstance(symbol, dict)
            ]

            loaded[path] = FileRecord(
                path=str(value.get("path", path)),
                size=int(value.get("size", 0)),
                modified_ns=int(value.get("modified_ns", 0)),
                sha256=str(value.get("sha256", "")),
                language=str(value.get("language", "")),
                symbols=symbols,
                imports=list(value.get("imports", [])),
                dependencies=list(value.get("dependencies", [])),
            )

        self.records = loaded

    def save(self) -> None:
        """Persist the current index atomically."""

        self.state_path.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": 1,
            "root": str(self.root),
            "updated_at": time.time(),
            "records": {
                path: asdict(record)
                for path, record in self.records.items()
            },
        }

        temporary = self.index_path.with_suffix(".tmp")

        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        temporary.replace(self.index_path)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def iter_files(self) -> Iterator[Path]:
        """Yield indexable project files."""

        for path in self.root.rglob("*"):
            if not path.is_file():
                continue

            relative_parts = path.relative_to(self.root).parts

            if any(part in self.ignored_dirs for part in relative_parts):
                continue

            if path.name in self.ignored_files:
                continue

            # Do not ingest arbitrary binary/huge files into model context.
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue

            yield path

    def relative_path(self, path: str | Path) -> str:
        """Return a stable POSIX-style project-relative path."""

        candidate = Path(path)

        if not candidate.is_absolute():
            candidate = self.root / candidate

        try:
            relative = candidate.resolve().relative_to(self.root)
        except ValueError as exc:
            raise ContextError(
                f"Path is outside the project: {candidate}"
            ) from exc

        return relative.as_posix()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build(self) -> dict[str, FileRecord]:
        """
        Fully scan and index the project.

        Existing records are reused when file metadata and hashes still
        match. This keeps repeated ASCS startups fast.
        """

        current_paths: set[str] = set()

        for path in self.iter_files():
            relative = self.relative_path(path)
            current_paths.add(relative)

            record = self._index_file(path, relative)
            self.records[relative] = record

        # Remove deleted files from the persistent index.
        stale = set(self.records) - current_paths

        for path in stale:
            del self.records[path]

        self._resolve_dependencies()
        self.save()

        return self.records

    def update(self) -> dict[str, FileRecord]:
        """Incrementally update the index from the current on-disk state.

        Only changed, new, or deleted files are processed:

        * new files are indexed,
        * existing files whose size/mtime still match are kept as-is,
        * deleted files are removed from the index.

        This avoids a full re-parse of an unchanged project on every run.
        Also re-resolves module dependencies (they may have shifted) and
        persists the result.
        """
        seen: set[str] = set()
        changed = False

        for path in self.iter_files():
            relative = self.relative_path(path)
            seen.add(relative)
            previous = self.records.get(relative)
            if previous is None:
                # New file: index it.
                record = self._index_file(path, relative)
                self.records[relative] = record
                changed = True
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if (
                previous.size == stat.st_size
                and previous.modified_ns == stat.st_mtime_ns
            ):
                continue  # fast path: unchanged
            # Content may have changed -> re-index (hash re-verified inside).
            record = self._index_file(path, relative)
            self.records[relative] = record
            changed = True

        # Remove deleted files.
        stale = set(self.records) - seen
        for path in stale:
            del self.records[path]
            changed = True

        if changed or stale:
            self._resolve_dependencies()
            self.save()

        return self.records

    def _index_file(self, path: Path, relative: str) -> FileRecord:
        try:
            stat = path.stat()
            raw = path.read_bytes()
        except OSError as exc:
            raise ContextError(f"Unable to read {path}: {exc}") from exc

        digest = hashlib.sha256(raw).hexdigest()

        previous = self.records.get(relative)

        if (
            previous
            and previous.size == stat.st_size
            and previous.modified_ns == stat.st_mtime_ns
            and previous.sha256 == digest
        ):
            return previous

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")

        language = self._language_for(path)

        symbols: list[Symbol] = []
        imports: list[str] = []

        if path.suffix.lower() in {".py", ".pyw"}:
            symbols, imports = self._parse_python(text)

        return FileRecord(
            path=relative,
            size=stat.st_size,
            modified_ns=stat.st_mtime_ns,
            sha256=digest,
            language=language,
            symbols=symbols,
            imports=imports,
        )

    def _language_for(self, path: Path) -> str:
        mapping = {
            ".py": "python",
            ".pyw": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".json": "json",
            ".toml": "toml",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sql": "sql",
            ".sh": "shell",
            ".bash": "shell",
            ".ps1": "powershell",
        }

        return mapping.get(path.suffix.lower(), "text")

    def _parse_python(self, text: str) -> tuple[list[Symbol], list[str]]:
        """Extract Python symbols and import dependencies safely."""

        try:
            tree = ast.parse(text)
        except SyntaxError:
            # A partially edited Python file should still be indexable.
            return self._fallback_python_symbols(text), []

        lines = text.splitlines()
        symbols: list[Symbol] = []
        imports: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(
                    Symbol(
                        name=node.name,
                        kind="function",
                        line_start=node.lineno,
                        line_end=getattr(
                            node,
                            "end_lineno",
                            node.lineno,
                        ),
                    )
                )

            elif isinstance(node, ast.ClassDef):
                symbols.append(
                    Symbol(
                        name=node.name,
                        kind="class",
                        line_start=node.lineno,
                        line_end=getattr(
                            node,
                            "end_lineno",
                            node.lineno,
                        ),
                    )
                )

            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        # Keep deterministic ordering.
        symbols.sort(key=lambda item: (item.line_start, item.name))
        imports = sorted(set(imports))

        # Avoid an unused-variable warning while making the fallback intent
        # obvious to readers.
        del lines

        return symbols, imports

    def _fallback_python_symbols(self, text: str) -> list[Symbol]:
        """Best-effort symbol extraction for temporarily invalid Python."""

        result: list[Symbol] = []

        pattern = re.compile(
            r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)",
            re.MULTILINE,
        )

        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            kind = "class" if "class" in match.group(0) else "function"

            result.append(
                Symbol(
                    name=match.group(1),
                    kind=kind,
                    line_start=line,
                    line_end=line,
                )
            )

        return result

    def _resolve_dependencies(self) -> None:
        """
        Resolve simple local Python imports into project-relative paths.

        External packages remain represented by their import name but are not
        treated as project dependencies.
        """

        python_files = {
            Path(path).with_suffix("").as_posix().replace("/", "."):
                path
            for path, record in self.records.items()
            if record.language == "python"
        }

        for record in self.records.values():
            dependencies: list[str] = []

            if record.language != "python":
                record.dependencies = dependencies
                continue

            current_package = Path(record.path).parent.as_posix().replace("/", ".")

            for imported in record.imports:
                candidates = [
                    imported,
                    (
                        f"{current_package}.{imported}"
                        if current_package not in ("", ".")
                        else imported
                    ),
                ]

                for candidate in candidates:
                    if candidate in python_files:
                        dependencies.append(python_files[candidate])
                        break

                    module_path = candidate.replace(".", "/")

                    for suffix in (".py", "/__init__.py"):
                        candidate_path = f"{module_path}{suffix}"

                        if candidate_path in self.records:
                            dependencies.append(candidate_path)
                            break

                    if dependencies and dependencies[-1] in self.records:
                        break

            record.dependencies = sorted(set(dependencies))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[FileRecord]:
        """
        Rank project files against a task/query.

        This is deliberately deterministic and dependency-aware. It does not
        pretend to be a vector database.
        """

        query_terms = self._terms(query)

        if not query_terms:
            return sorted(
                self.records.values(),
                key=lambda record: record.path,
            )[:limit]

        scored: list[tuple[int, FileRecord]] = []

        for record in self.records.values():
            score = self._score_record(record, query_terms)

            if score > 0:
                scored.append((score, record))

        scored.sort(
            key=lambda pair: (-pair[0], pair[1].path),
        )

        primary = [record for _, record in scored[:limit]]

        # Add direct dependencies of highly relevant files.
        seen = {record.path for record in primary}

        for record in list(primary):
            for dependency in record.dependencies:
                if dependency in self.records and dependency not in seen:
                    primary.append(self.records[dependency])
                    seen.add(dependency)

                    if len(primary) >= limit:
                        return primary

        return primary

    def _score_record(
        self,
        record: FileRecord,
        terms: set[str],
    ) -> int:
        path_text = record.path.lower()
        symbol_text = " ".join(symbol.name for symbol in record.symbols).lower()
        import_text = " ".join(record.imports).lower()

        score = 0

        for term in terms:
            if term in path_text:
                score += 8

            if term in symbol_text:
                score += 6

            if term in import_text:
                score += 2

        # Small bonus for Python source when the task appears code-oriented.
        if record.language == "python":
            score += 1

        return score

    def _terms(self, query: str) -> set[str]:
        return {
            term.lower()
            for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)
        }

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunks_for_file(
        self,
        path: str | Path,
        *,
        max_tokens: int = DEFAULT_CHUNK_TOKENS,
    ) -> list[ContextChunk]:
        """
        Split a file into model-sized chunks while respecting source
        boundaries where possible.
        """

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        relative = self.relative_path(path)
        absolute = self.root / relative

        try:
            text = absolute.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = absolute.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            raise ContextError(
                f"Unable to read {relative}: {exc}"
            ) from exc

        lines = text.splitlines()

        if not lines:
            return [
                ContextChunk(
                    chunk_id=self._chunk_id(relative, 1, 1),
                    path=relative,
                    start_line=1,
                    end_line=1,
                    text="",
                    estimated_tokens=0,
                )
            ]

        record = self.records.get(relative)

        if record and record.symbols:
            return self._chunk_using_symbols(
                relative,
                lines,
                record.symbols,
                max_tokens,
            )

        return self._chunk_by_lines(
            relative,
            lines,
            max_tokens,
        )

    def _chunk_using_symbols(
        self,
        relative: str,
        lines: list[str],
        symbols: list[Symbol],
        max_tokens: int,
    ) -> list[ContextChunk]:
        chunks: list[ContextChunk] = []

        # Build logical blocks from symbol boundaries.
        boundaries: list[tuple[int, int, str | None]] = []

        cursor = 1

        for symbol in symbols:
            start = max(1, symbol.line_start)

            if start > cursor:
                boundaries.append((cursor, start - 1, None))

            end = min(len(lines), max(start, symbol.line_end))

            boundaries.append((start, end, symbol.name))
            cursor = end + 1

        if cursor <= len(lines):
            boundaries.append((cursor, len(lines), None))

        current_lines: list[str] = []
        current_start = 1
        current_symbols: list[str] = []

        def flush(end_line: int) -> None:
            nonlocal current_lines, current_start, current_symbols

            if not current_lines:
                return

            text = "\n".join(current_lines)
            chunks.append(
                ContextChunk(
                    chunk_id=self._chunk_id(
                        relative,
                        current_start,
                        end_line,
                    ),
                    path=relative,
                    start_line=current_start,
                    end_line=end_line,
                    text=text,
                    estimated_tokens=self.estimate_tokens(text),
                    symbols=list(current_symbols),
                )
            )

            current_lines = []
            current_symbols = []

        for start, end, symbol_name in boundaries:
            block = lines[start - 1 : end]
            block_text = "\n".join(block)

            if (
                current_lines
                and self.estimate_tokens(
                    "\n".join(current_lines + block)
                )
                > max_tokens
            ):
                flush(start - 1)
                current_start = start

            if self.estimate_tokens(block_text) > max_tokens:
                # A single enormous function/class still needs splitting.
                flush(start - 1)

                oversized = self._chunk_by_lines(
                    relative,
                    block,
                    max_tokens,
                    start_line=start,
                )

                chunks.extend(oversized)
                current_start = end + 1
                continue

            if not current_lines:
                current_start = start

            current_lines.extend(block)

            if symbol_name:
                current_symbols.append(symbol_name)

        flush(len(lines))

        return self._merge_tiny_chunks(chunks, max_tokens)

    def _chunk_by_lines(
        self,
        relative: str,
        lines: list[str],
        max_tokens: int,
        *,
        start_line: int = 1,
    ) -> list[ContextChunk]:
        chunks: list[ContextChunk] = []

        current: list[str] = []
        current_start = start_line

        for offset, line in enumerate(lines):
            absolute_line = start_line + offset

            candidate = "\n".join(current + [line])

            if current and self.estimate_tokens(candidate) > max_tokens:
                text = "\n".join(current)

                chunks.append(
                    ContextChunk(
                        chunk_id=self._chunk_id(
                            relative,
                            current_start,
                            absolute_line - 1,
                        ),
                        path=relative,
                        start_line=current_start,
                        end_line=absolute_line - 1,
                        text=text,
                        estimated_tokens=self.estimate_tokens(text),
                    )
                )

                current = []
                current_start = absolute_line

            current.append(line)

        if current:
            text = "\n".join(current)

            chunks.append(
                ContextChunk(
                    chunk_id=self._chunk_id(
                        relative,
                        current_start,
                        start_line + len(lines) - 1,
                    ),
                    path=relative,
                    start_line=current_start,
                    end_line=start_line + len(lines) - 1,
                    text=text,
                    estimated_tokens=self.estimate_tokens(text),
                )
            )

        return chunks

    def _merge_tiny_chunks(
        self,
        chunks: list[ContextChunk],
        max_tokens: int,
    ) -> list[ContextChunk]:
        """
        Merge adjacent tiny chunks when doing so remains inside the budget.

        This reduces needless context fragmentation.
        """

        if len(chunks) < 2:
            return chunks

        merged: list[ContextChunk] = []

        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue

            previous = merged[-1]

            combined = previous.text + "\n" + chunk.text

            if (
                previous.estimated_tokens < max_tokens // 4
                and self.estimate_tokens(combined) <= max_tokens
                and previous.path == chunk.path
                and previous.end_line + 1 >= chunk.start_line
            ):
                merged[-1] = ContextChunk(
                    chunk_id=self._chunk_id(
                        previous.path,
                        previous.start_line,
                        chunk.end_line,
                    ),
                    path=previous.path,
                    start_line=previous.start_line,
                    end_line=chunk.end_line,
                    text=combined,
                    estimated_tokens=self.estimate_tokens(combined),
                    symbols=previous.symbols + chunk.symbols,
                    related_files=previous.related_files
                    + chunk.related_files,
                )
            else:
                merged.append(chunk)

        return merged

    # ------------------------------------------------------------------
    # Task context assembly
    # ------------------------------------------------------------------

    def build_context(
        self,
        task: str,
        *,
        max_tokens: int = DEFAULT_CHUNK_TOKENS,
        max_files: int = 16,
    ) -> ContextBundle:
        """
        Build a single model-ready context bundle.

        The returned context is always budgeted to ``max_tokens`` or below
        according to ASCS's conservative token estimator.
        """

        if not task.strip():
            raise ValueError("task must not be empty")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        records = self.search(task, limit=max_files)

        chunks: list[ContextChunk] = []
        used_tokens = 0
        seen_chunks: set[str] = set()

        for record in records:
            file_chunks = self.chunks_for_file(
                record.path,
                max_tokens=max_tokens,
            )

            for chunk in file_chunks:
                if chunk.chunk_id in seen_chunks:
                    continue

                # Account for a small wrapper/header around each chunk.
                overhead = self.estimate_tokens(
                    f"===== {chunk.path}:{chunk.start_line}-{chunk.end_line} ====="
                )

                required = chunk.estimated_tokens + overhead

                if used_tokens + required > max_tokens:
                    continue

                chunk.related_files = list(record.dependencies)

                chunks.append(chunk)
                seen_chunks.add(chunk.chunk_id)
                used_tokens += required

                if used_tokens >= max_tokens:
                    break

            if used_tokens >= max_tokens:
                break

        return ContextBundle(
            task=task,
            chunks=chunks,
            estimated_tokens=used_tokens,
            files=sorted({chunk.path for chunk in chunks}),
        )

    # ------------------------------------------------------------------
    # Retrieval (hierarchical)
    # ------------------------------------------------------------------

    def file_summary(self, path: str | Path) -> str:
        """Return a short, deterministic Level-2 summary of one file."""
        relative = self.relative_path(path)
        record = self.records.get(relative)
        if record is None:
            return f"{relative}: not indexed"
        language = record.language or "text"
        symbol_names = ", ".join(
            f"{s.kind} {s.name}" for s in record.symbols[:12]
        )
        if record.symbols:
            symbol_text = f"; symbols: {symbol_names}"
        else:
            symbol_text = ""
        imports_text = (
            f"; imports {len(record.imports)} module(s)"
            if record.imports
            else ""
        )
        return (
            f"{relative} [{language}, {record.size} bytes, "
            f"{len(record.symbols)} symbol(s)]{symbol_text}{imports_text}"
        )

    def file_summaries(self, limit: int = 200) -> list[str]:
        """Level-2 directory/file summaries for understanding architecture."""
        ordered = sorted(self.records.values(), key=lambda r: r.path)
        return [self.file_summary(record.path) for record in ordered[:limit]]

    def dependents(self, path: str | Path) -> list[str]:
        """Files that import ``path`` (reverse dependency edges)."""
        target = self.relative_path(path)
        result: list[str] = []
        for record in self.records.values():
            if target in record.dependencies:
                result.append(record.path)
        return sorted(set(result))

    def related_files(self, path: str | Path, *, limit: int = 20) -> list[str]:
        """Dependency-aware related files for ``path``.

        Returns the union of the file's own dependencies and its dependents
        (importers), then any tests whose name matches the module, capped at
        ``limit``.
        """
        relative = self.relative_path(path)
        record = self.records.get(relative)
        related: set[str] = set()
        if record is not None:
            related.update(record.dependencies)
        related.update(self.dependents(relative))

        # Tests targeting this module: tests/<module>_test.py or test_<module>.py
        stem = Path(relative).stem
        module_basename = Path(relative).name.split(".")[0]
        for candidate in self.records:
            candidate_name = Path(candidate).name.lower()
            if candidate_name in {
                f"test_{module_basename}.py",
                f"{module_basename}_test.py",
                f"test_{module_basename.lower()}.py",
                f"{module_basename.lower()}_test.py",
            } and not candidate.endswith(".pyc"):
                related.add(candidate)
            elif (
                stem in candidate
                and ("tests/" in candidate or candidate.startswith("test_"))
            ):
                related.add(candidate)

        ordered = sorted(related, key=lambda p: (p != relative, p))
        return ordered[:limit]

    def retrieve(
        self,
        task: str,
        *,
        level: int = 3,
        max_tokens: int = DEFAULT_CHUNK_TOKENS,
        max_files: int = 16,
    ) -> ContextBundle:
        """Hierarchical retrieval for a task.

        Levels select how much surrounding structure is included:

        * ``level=1`` - project metadata only (callers should pair this with
          :meth:`ProjectStore.refresh` to obtain the manifest).
        * ``level=2`` - directory/file summaries of the most relevant files.
        * ``level=3`` - relevant source files, chunked to budget (default).
        * ``level=4`` - exact code regions + related (dependency/tests) files.
        """
        if level < 1 or level > 4:
            raise ValueError("level must be in 1..4")

        bundle = self.build_context(task, max_tokens=max_tokens, max_files=max_files)

        if level == 1:
            return ContextBundle(
                task=task,
                chunks=[],
                estimated_tokens=0,
                files=list(self._level1_files()),
            )

        if level == 2:
            summaries = self.file_summaries(limit=max_files * 2)
            return ContextBundle(
                task=task,
                chunks=[
                    ContextChunk(
                        chunk_id=self._chunk_id("summaries", idx, idx),
                        path="<summaries>",
                        start_line=0,
                        end_line=0,
                        text=summary,
                        estimated_tokens=self.estimate_tokens(summary),
                        language="",
                    )
                    for idx, summary in enumerate(summaries)
                ],
                estimated_tokens=sum(
                    self.estimate_tokens(s) for s in summaries
                ),
                files=sorted({r.path for r in self.search(task, limit=max_files)}),
            )

        # level 3 or 4: annotate chunks with language + related files.
        for chunk in bundle.chunks:
            record = self.records.get(chunk.path)
            if record is not None:
                chunk.language = record.language
                if level >= 4:
                    chunk.related_files = (
                        chunk.related_files or self.related_files(chunk.path)
                    )

        return bundle

    def _level1_files(self) -> list[str]:
        """Small always-available metadata list (used with level=1)."""
        return ["(project manifest lives in .ascs/project_manifest.json)"]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count conservatively.

        This intentionally does not claim to reproduce a particular model's
        tokenizer. It is a safety budget used before sending context to Ollama.
        """

        if not text:
            return 0

        # Character estimate plus a small whitespace/punctuation correction.
        return max(
            1,
            int(len(text) / _TOKEN_CHARS),
        )

    @staticmethod
    def _chunk_id(path: str, start: int, end: int) -> str:
        value = f"{path}:{start}:{end}".encode("utf-8")
        return hashlib.sha1(value).hexdigest()[:16]


def git_status(root: str | Path) -> str:
    """Return a compact Git status for context/debugging."""

    project = Path(root).resolve()

    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""

    return result.stdout.strip()


def create_project_index(
    root: str | Path,
    *,
    ignored_dirs: Iterable[str] | None = None,
    ignored_files: Iterable[str] | None = None,
) -> ProjectIndex:
    """Create and fully build a project index."""

    index = ProjectIndex(
        root,
        ignored_dirs=ignored_dirs,
        ignored_files=ignored_files,
    )

    index.build()
    return index