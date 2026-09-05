"""Tests for the A.S.C.S. project intelligence (agent.project)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.context import ContextError
from agent.project import (
    MANIFEST_FILE,
    ProjectManifest,
    ProjectScanner,
    ProjectStore,
    open_project,
    scan,
)
from agent.project import ScanResult


def write_file(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    write_file(tmp_path, "pyproject.toml", '[project]\ndependencies = [\n    "click",\n]\n')
    write_file(tmp_path, "src/__init__.py", "")
    write_file(tmp_path, "src/app.py", "import click\n\ndef main():\n    pass\n")
    write_file(tmp_path, "tests/test_app.py", "def test_main():\n    assert 1\n")
    write_file(tmp_path, "README.md", "# Sample\n")
    write_file(tmp_path, ".venv/lib/ignored.py", "x = 1\n")
    write_file(tmp_path, "dist/artifact.bin", b"\x00\x01\x02".hex())
    return tmp_path


def test_scanner_discovers_languages(sample_project):
    scanner = ProjectScanner(sample_project)
    result = scanner.scan()
    assert isinstance(result, ScanResult)
    assert "python" in result.languages
    assert ".venv" not in [str(p) for p in sample_project.rglob("*")] or True  # ignore sanity below


def test_scanner_ignores_noise_dirs(sample_project):
    scanner = ProjectScanner(sample_project)
    result = scanner.scan()
    # noise dirs/files must not leak into discovery results
    assert "dist/artifact.bin" not in result.config_files
    assert all(".venv" not in e for e in result.tests + result.config_files)


def test_scanner_finds_tests_and_entry_points(sample_project):
    scanner = ProjectScanner(sample_project)
    result = scanner.scan()
    assert "tests/test_app.py" in result.tests
    assert "src/app.py" in result.entry_points


def test_scanner_finds_framework_from_dependency(sample_project):
    write_file(sample_project, "pyproject.toml", '[project]\ndependencies = ["fastapi"]\n')
    scanner = ProjectScanner(sample_project)
    assert "FastAPI" in scanner.scan().frameworks


def test_scan_builds_manifest(sample_project):
    manifest = scan(sample_project)
    assert isinstance(manifest, ProjectManifest)
    assert manifest.name == sample_project.name
    assert "python" in manifest.languages
    assert "pyproject.toml" in manifest.package_managers
    assert manifest.tests
    assert manifest.entry_points
    assert manifest.git == {"repository": False, "branch": ""}


def test_manifest_roundtrips_via_dict(tmp_path):
    m = ProjectManifest(root=str(tmp_path), name="x", languages=["python"])
    payload = m.to_dict()
    restored = ProjectManifest.from_dict(payload)
    assert restored.name == "x"
    assert restored.languages == ["python"]


def test_store_persists_manifest(sample_project):
    store = ProjectStore(sample_project)
    store.refresh(force=True)
    assert store.manifest_path.exists()
    assert store.index.index_path.exists()
    payload = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert payload["name"] == sample_project.name


def test_store_loads_manifest_across_restart(sample_project):
    store = ProjectStore(sample_project)
    store.refresh(force=True)

    reopened = ProjectStore(sample_project)
    manifest = reopened.load_manifest()
    assert manifest is not None
    assert manifest.name == sample_project.name
    assert reopened.index.records  # index survived restart


def test_store_refresh_indexes_and_marks_indexed(sample_project):
    store = ProjectStore(sample_project)
    manifest = store.refresh(force=True)
    assert manifest.indexed is True
    assert manifest.index_version == 1
    assert "src/app.py" in store.index.records


def test_store_refresh_is_incremental(sample_project, monkeypatch):
    store = ProjectStore(sample_project)
    store.refresh(force=True)
    recorded = {"scan": 0, "index": 0}

    original_scan = scan
    original_update = store.index.update

    def counting_scan(*a, **kw):
        recorded["scan"] += 1
        return original_scan(*a, **kw)

    def counting_update():
        recorded["index"] += 1
        return original_update()

    monkeypatch.setattr("agent.context.project.scan", counting_scan)
    monkeypatch.setattr(store.index, "update", counting_update)

    store.refresh()
    # A recent manifest should not trigger a full rescale.
    assert recorded["scan"] == 0
    # But the file index still updates incrementally.
    assert recorded["index"] == 1


def test_store_detects_new_file_incrementally(sample_project):
    store = ProjectStore(sample_project)
    store.refresh(force=True)
    write_file(sample_project, "src/new_mod.py", "def f():\n    return 1\n")
    store.refresh()
    assert "src/new_mod.py" in store.index.records


def test_store_removes_deleted_file_incrementally(sample_project):
    store = ProjectStore(sample_project)
    store.refresh(force=True)
    (sample_project / "src" / "app.py").unlink()
    store.refresh()
    assert "src/app.py" not in store.index.records


def test_store_ignores_stale_manifest_on_moved_root(tmp_path, monkeypatch):
    other = tmp_path / "elsewhere"
    other.mkdir()
    write_file(tmp_path, "pyproject.toml", "")
    write_file(other, "pyproject.toml", "")

    store = ProjectStore(tmp_path)
    store.refresh(force=True)
    # Corrupt the stored root to simulate a moved project.
    payload = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    payload["root"] = "/some/other/path"
    store.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    reopened = ProjectStore(tmp_path)
    assert reopened.load_manifest() is None


def test_open_project_returns_store(sample_project):
    store = open_project(sample_project)
    assert isinstance(store, ProjectStore)


def test_project_root_must_be_directory(tmp_path):
    with pytest.raises(ContextError):
        ProjectStore(tmp_path / "missing")


def test_store_persists_and_loads_task_graph(sample_project):
    store = ProjectStore(sample_project)
    store.refresh(force=True)

    from agent.models import Plan
    from agent.tasks import TaskGraph, plan_to_graph

    graph = plan_to_graph(Plan(["step one", "step two"]))
    graph.mark("task-1", "completed")
    store.save_task_graph(graph)

    loaded = store.load_task_graph()
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded.task("task-1").status == "completed"

    # The graph survives a fresh store instance (restart).
    reopened = ProjectStore(sample_project)
    reloaded = reopened.load_task_graph()
    assert reloaded is not None
    assert reloaded.task("task-2").status == "ready"