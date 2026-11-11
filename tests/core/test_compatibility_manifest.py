"""The compatibility manifest must match the shipped package version."""

from __future__ import annotations

import json
from pathlib import Path

from agent import __version__

MANIFEST = Path(__file__).resolve().parents[2] / "compatibility.json"


def test_manifest_matches_package_version():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["project"] == "ASCS"
    assert data["version"] == __version__
    assert data["standalone"] is True
    assert data["requires_core"] is False
    assert data["core_integration"]["optional"] is True


def test_manifest_declares_required_platforms():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    systems = {entry["os"]: set(entry["architectures"]) for entry in data["supported_operating_systems"]}
    assert systems["windows"] >= {"x86_64", "arm64"}
    assert systems["macos"] == {"x86_64", "arm64"}
    assert systems["linux"] >= {"x86_64"}
