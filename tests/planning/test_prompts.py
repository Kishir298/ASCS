"""Direct tests for prompt construction (agent.prompts)."""

from __future__ import annotations

import pytest

from agent.config import AgentConfig
from agent.planning.prompts import (
    _environment_text,
    _mode_instructions,
    malformed_feedback,
    system_prompt,
    task_message,
    tool_error_feedback,
)


def test_system_prompt_mode_auto(config):
    text = system_prompt(config)
    assert "A.S.C.S." in text
    assert "MODE: AUTO" in text
    assert "RESPONSE CONTRACT" in text


def test_system_prompt_plan_mode(config):
    plan = AgentConfig(workspace=config.workspace, mode="PLAN")
    text = system_prompt(plan)
    assert "MODE: PLAN" in text
    assert "must NOT modify" in text


def test_system_prompt_build_mode(config):
    build = AgentConfig(workspace=config.workspace, mode="BUILD")
    text = system_prompt(build)
    assert "MODE: BUILD" in text


def test_system_prompt_includes_project_block(config):
    text = system_prompt(config, project="- Languages: python\n- Tests: pytest\n")
    assert "PROJECT INTELLIGENCE" in text
    assert "python" in text
    assert "pytest" in text


def test_system_prompt_empty_project_block_omitted(config):
    text = system_prompt(config, project=None)
    assert "PROJECT INTELLIGENCE" not in text
    text2 = system_prompt(config, project="   ")
    assert "PROJECT INTELLIGENCE" not in text2


def test_environment_text_lists_host_facts():
    text = _environment_text()
    assert "ENVIRONMENT" in text
    assert "Operating system" in text


def test_task_message_wraps_task():
    msg = task_message("Do the thing")
    assert msg == {"role": "user", "content": "Do the thing"}


def test_malformed_feedback_guides_json():
    msg = malformed_feedback("no JSON found")
    assert msg["role"] == "user"
    assert "rejected" in msg["content"]
    assert "JSON" in msg["content"]


def test_tool_error_feedback_reference_tool():
    msg = tool_error_feedback("write_file", "disabled in PLAN mode")
    assert msg["role"] == "user"
    assert "write_file" in msg["content"]
    assert "disabled" in msg["content"]


def test_mode_instructions_for_safe_build(config):
    safe = AgentConfig(workspace=config.workspace, mode="BUILD", approval=True)
    text = _mode_instructions(safe)
    assert "approval" in text.lower()


def test_system_prompt_lists_enabled_tools(config):
    text = system_prompt(config)
    # A modifying tool present in AUTO
    assert "run_command" in text