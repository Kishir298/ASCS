"""Tests for the A.S.C.S. persistent project context engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.context import (
    DEFAULT_CHUNK_TOKENS,
    ContextError,
    ProjectIndex,
    create_project_index,
    git_status,
)


def write_file(root: Path, relative: str, content: str) -> Path:
    """Create a UTF-8 test file and return its path."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Create a small representative Python project."""
    write_file(
        tmp_path,
        "main.py",
        """from services.calculator import Calculator


def main():
    calculator = Calculator()
    return calculator.add(2, 3)


if __name__ == "__main__":
    main()
""",
    )

    write_file(
        tmp_path,
        "services/__init__.py",
        "",
    )

    write_file(
        tmp_path,
        "services/calculator.py",
        """class Calculator:
    def add(self, left, right):
        return left + right

    def subtract(self, left, right):
        return left - right
""",
    )

    write_file(
        tmp_path,
        "README.md",
        """# Example Project

This project contains a calculator service.
""",
    )

    return tmp_path


def test_project_index_builds_and_persists(project: Path) -> None:
    index = ProjectIndex(project)

    records = index.build()

    assert "main.py" in records
    assert "services/calculator.py" in records
    assert "README.md" in records

    assert index.index_path.exists()

    payload = json.loads(
        index.index_path.read_text(encoding="utf-8")
    )

    assert payload["version"] == 1
    assert "main.py" in payload["records"]


def test_project_index_loads_existing_index(project: Path) -> None:
    first = ProjectIndex(project)
    first.build()

    second = ProjectIndex(project)

    assert set(second.records) == set(first.records)
    assert second.records["main.py"].sha256 == (
        first.records["main.py"].sha256
    )


def test_ignored_directories_are_not_indexed(project: Path) -> None:
    write_file(
        project,
        ".git/config.txt",
        "should not be indexed",
    )

    write_file(
        project,
        ".venv/test.py",
        "should not be indexed",
    )

    write_file(
        project,
        "__pycache__/cache.py",
        "should not be indexed",
    )

    write_file(
        project,
        "node_modules/package.js",
        "should not be indexed",
    )

    index = ProjectIndex(project)
    records = index.build()

    assert ".git/config.txt" not in records
    assert ".venv/test.py" not in records
    assert "__pycache__/cache.py" not in records
    assert "node_modules/package.js" not in records


def test_python_symbols_are_extracted(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    record = index.records["services/calculator.py"]

    symbols = {
        (symbol.name, symbol.kind)
        for symbol in record.symbols
    }

    assert ("Calculator", "class") in symbols
    assert ("add", "function") in symbols
    assert ("subtract", "function") in symbols


def test_python_imports_are_extracted(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    record = index.records["main.py"]

    assert "services.calculator" in record.imports


def test_local_python_dependencies_are_resolved(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    record = index.records["main.py"]

    assert "services/calculator.py" in record.dependencies


def test_search_finds_relevant_files(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    results = index.search(
        "calculator add service",
        limit=5,
    )

    paths = [record.path for record in results]

    assert "services/calculator.py" in paths


def test_search_is_deterministic(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    first = [record.path for record in index.search("calculator")]
    second = [record.path for record in index.search("calculator")]

    assert first == second


def test_search_includes_direct_dependencies(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    results = index.search(
        "main calculator",
        limit=5,
    )

    paths = [record.path for record in results]

    assert "main.py" in paths
    assert "services/calculator.py" in paths


def test_chunks_respect_token_budget(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    chunks = index.chunks_for_file(
        "services/calculator.py",
        max_tokens=20,
    )

    assert chunks

    for chunk in chunks:
        assert chunk.estimated_tokens <= 20


def test_large_file_is_split_into_multiple_chunks(
    project: Path,
) -> None:
    lines = [
        f"def function_{number}():"
        for number in range(100)
    ]

    lines.extend(
        "    return " + str(number)
        for number in range(100)
    )

    write_file(
        project,
        "large.py",
        "\n".join(lines),
    )

    index = ProjectIndex(project)
    index.build()

    chunks = index.chunks_for_file(
        "large.py",
        max_tokens=30,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.estimated_tokens <= 30


def test_context_bundle_respects_requested_budget(
    project: Path,
) -> None:
    index = ProjectIndex(project)
    index.build()

    bundle = index.build_context(
        "calculator service add",
        max_tokens=100,
    )

    assert bundle.estimated_tokens <= 100
    assert bundle.files

    for chunk in bundle.chunks:
        assert chunk.estimated_tokens >= 0


def test_context_bundle_contains_file_headers(
    project: Path,
) -> None:
    index = ProjectIndex(project)
    index.build()

    bundle = index.build_context(
        "calculator",
        max_tokens=200,
    )

    assert bundle.text

    for path in bundle.files:
        assert path in bundle.text


def test_default_chunk_budget_is_4096() -> None:
    assert DEFAULT_CHUNK_TOKENS == 4096


def test_empty_query_returns_deterministic_records(
    project: Path,
) -> None:
    index = ProjectIndex(project)
    index.build()

    results = index.search("", limit=2)

    assert len(results) == 2
    assert results[0].path == "README.md"


def test_relative_path_is_stable(project: Path) -> None:
    index = ProjectIndex(project)

    absolute = project / "main.py"

    assert index.relative_path(absolute) == "main.py"
    assert index.relative_path("main.py") == "main.py"


def test_relative_path_rejects_paths_outside_project(
    project: Path,
) -> None:
    index = ProjectIndex(project)

    outside = project.parent / "outside.py"

    with pytest.raises(ContextError):
        index.relative_path(outside)


def test_missing_project_root_is_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    with pytest.raises(ContextError):
        ProjectIndex(missing)


def test_project_root_must_be_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "project.txt"
    file_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ContextError):
        ProjectIndex(file_path)


def test_invalid_python_still_gets_fallback_symbols(
    project: Path,
) -> None:
    write_file(
        project,
        "broken.py",
        """def first():
    return 1

def second(
    return 2
""",
    )

    index = ProjectIndex(project)
    index.build()

    record = index.records["broken.py"]

    names = {symbol.name for symbol in record.symbols}

    assert "first" in names


def test_empty_file_produces_one_empty_chunk(
    project: Path,
) -> None:
    write_file(project, "empty.py", "")

    index = ProjectIndex(project)
    index.build()

    chunks = index.chunks_for_file("empty.py")

    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 1
    assert chunks[0].text == ""
    assert chunks[0].estimated_tokens == 0


def test_estimate_tokens_is_zero_for_empty_text() -> None:
    assert ProjectIndex.estimate_tokens("") == 0


def test_estimate_tokens_is_at_least_one_for_nonempty_text() -> None:
    assert ProjectIndex.estimate_tokens("x") == 1


def test_build_removes_deleted_files(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    assert "README.md" in index.records

    (project / "README.md").unlink()

    index.build()

    assert "README.md" not in index.records


def test_changed_file_is_reindexed(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    original_hash = index.records["main.py"].sha256

    write_file(
        project,
        "main.py",
        """def changed():
    return True
""",
    )

    index.build()

    assert index.records["main.py"].sha256 != original_hash

    names = {
        symbol.name
        for symbol in index.records["main.py"].symbols
    }

    assert "changed" in names


def test_create_project_index_builds_immediately(
    project: Path,
) -> None:
    index = create_project_index(project)

    assert index.records
    assert index.index_path.exists()


def test_git_status_returns_string(project: Path) -> None:
    result = git_status(project)

    assert isinstance(result, str)


def test_custom_ignored_directory(project: Path) -> None:
    write_file(
        project,
        "generated/generated.py",
        "def generated():\n    pass\n",
    )

    index = ProjectIndex(
        project,
        ignored_dirs={"generated"},
    )
    index.build()

    assert "generated/generated.py" not in index.records


def test_custom_ignored_file(project: Path) -> None:
    write_file(
        project,
        "secret.txt",
        "do not index",
    )

    index = ProjectIndex(
        project,
        ignored_files={"secret.txt"},
    )
    index.build()

    assert "secret.txt" not in index.records


def test_save_is_valid_json(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    payload = json.loads(
        index.index_path.read_text(encoding="utf-8")
    )

    assert isinstance(payload, dict)
    assert isinstance(payload["records"], dict)


def test_chunk_ids_are_stable(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    first = index.chunks_for_file(
        "services/calculator.py",
        max_tokens=50,
    )

    second = index.chunks_for_file(
        "services/calculator.py",
        max_tokens=50,
    )

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_chunk_line_ranges_are_valid(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    chunks = index.chunks_for_file(
        "services/calculator.py",
        max_tokens=50,
    )

    for chunk in chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_context_bundle_records_task(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    bundle = index.build_context(
        "find the calculator",
        max_tokens=100,
    )

    assert bundle.task == "find the calculator"


def test_context_bundle_generated_at_is_numeric(
    project: Path,
) -> None:
    index = ProjectIndex(project)
    index.build()

    bundle = index.build_context(
        "calculator",
        max_tokens=100,
    )

    assert isinstance(bundle.generated_at, float)


def test_project_index_handles_non_utf8_text(
    project: Path,
) -> None:
    path = project / "weird.txt"
    path.write_bytes(b"hello \xff world")

    index = ProjectIndex(project)
    index.build()

    assert "weird.txt" in index.records


def test_non_indexable_binary_extension_is_ignored(
    project: Path,
) -> None:
    path = project / "image.bin"
    path.write_bytes(b"\x00\x01\x02\x03")

    index = ProjectIndex(project)
    index.build()

    assert "image.bin" not in index.records


def test_invalid_chunk_budget_is_rejected(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    with pytest.raises(ValueError):
        index.chunks_for_file(
            "main.py",
            max_tokens=0,
        )


def test_invalid_context_budget_is_rejected(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    with pytest.raises(ValueError):
        index.build_context(
            "calculator",
            max_tokens=0,
        )


def test_empty_context_task_is_rejected(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    with pytest.raises(ValueError):
        index.build_context(
            "",
            max_tokens=100,
        )


def test_search_limit_is_respected(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    results = index.search(
        "python calculator",
        limit=1,
    )

    assert len(results) <= 1


def test_context_files_are_unique(project: Path) -> None:
    index = ProjectIndex(project)
    index.build()

    bundle = index.build_context(
        "calculator service",
        max_tokens=300,
    )

    assert len(bundle.files) == len(set(bundle.files))
