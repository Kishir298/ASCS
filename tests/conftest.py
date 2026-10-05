"""Shared fixtures for the coding-agent test suite.

Runs against a temporary workspace so tests never touch real projects.
"""

from __future__ import annotations

import os
import pytest

from agent.config import AgentConfig
from agent.workspace import Workspace


# External validation markers — skipped unless RISALIVE=1
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: requires live Ollama model (RISALIVE=1)"
    )

def pytest_collection_modifyitems(config, items):
    if os.environ.get("RISALIVE") != "1":
        skip_live = pytest.mark.skip(reason="RISALIVE=1 not set")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)


@pytest.fixture
def workspace(tmp_path):
    """A Workspace rooted at a fresh temp directory."""
    return Workspace(tmp_path)


@pytest.fixture
def ws_root(tmp_path):
    """The raw pathlib root of a fresh temp workspace."""
    return tmp_path


@pytest.fixture
def config(tmp_path):
    """A minimal AUTO-mode config pointing at a temp workspace."""
    return AgentConfig(workspace=tmp_path, mode="AUTO")


def run_in_workspace(cfg, **overrides):
    """Create a config with overrides and a corresponding Workspace."""
    merged = cfg
    if overrides:
        merged = cfg.__class__(
            **{**{f: getattr(cfg, f) for f in cfg.__dataclass_fields__}, **overrides}
        )
    return merged
