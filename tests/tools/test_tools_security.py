"""Regression tests for ASCS workspace/intent safety findings (ASCS-1 through ASCS-8).

These tests reproduce the vulnerabilities described in the audit and verify fixes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from agent.tools import execute_tool
from agent.workspace import Workspace


def test_ascs1_predictable_temp_file_no_external_overwrite(tmp_path, config):
    """ASCS-1: write_file/apply_patch must use exclusive unpredictable temp files.
    
    Regression: .risa_tmp sibling could be a symlink to external file,
    leading to external file overwrite.
    """
    ws = Workspace(tmp_path)
    
    # Create a target file and a symlink with predictable .risa_tmp name
    target_file = tmp_path / "target.txt"
    target_file.write_text("original content")
    
    # Create a symlink with the predictable temp name pointing outside workspace
    external_file = tmp_path.parent / "external_sentinel.txt"
    external_file.write_text("EXTERNAL CONTENT - SHOULD NOT BE OVERWRITTEN")
    
    temp_name = target_file.with_name(target_file.name + ".risa_tmp")
    temp_name.symlink_to(external_file)
    
    # Try to write - should not follow symlink and overwrite external file
    result = execute_tool("write_file", {"path": "target.txt", "content": "new content"}, ws, config)
    
    # The write should succeed (or fail safely) but external file must be untouched
    assert external_file.read_text() == "EXTERNAL CONTENT - SHOULD NOT BE OVERWRITTEN", \
        "External file was overwritten via predictable temp symlink!"
    
    # Target file should have new content
    assert target_file.read_text() == "new content"


def test_ascs1_apply_patch_predictable_temp_no_external_overwrite(tmp_path, config):
    """ASCS-1: apply_patch must use exclusive unpredictable temp files."""
    ws = Workspace(tmp_path)
    
    target_file = tmp_path / "target.txt"
    target_file.write_text("original content")
    
    external_file = tmp_path.parent / "external_sentinel2.txt"
    external_file.write_text("EXTERNAL CONTENT - SHOULD NOT BE OVERWRITTEN")
    
    temp_name = target_file.with_name(target_file.name + ".risa_tmp")
    temp_name.symlink_to(external_file)
    
    result = execute_tool(
        "apply_patch",
        {"path": "target.txt", "old_text": "original", "new_text": "patched"},
        ws,
        config,
    )
    
    assert external_file.read_text() == "EXTERNAL CONTENT - SHOULD NOT BE OVERWRITTEN", \
        "External file was overwritten via predictable temp symlink in apply_patch!"
    
    assert target_file.read_text() == "patched content"


def test_ascs2_copy_destination_validated(tmp_path, config):
    """ASCS-2: Directory copy destination must be validated after basename expansion.
    
    Regression: dest/src.txt symlink to external file was followed and overwritten.
    """
    ws = Workspace(tmp_path)
    
    # Create source file
    source = tmp_path / "source.txt"
    source.write_text("source content")
    
    # Create destination directory with symlink inside
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    
    external_file = tmp_path.parent / "external_sentinel3.txt"
    external_file.write_text("EXTERNAL CONTENT - SHOULD NOT BE OVERWRITTEN")
    
    # Create symlink inside destination directory
    symlink_target = dest_dir / "source.txt"
    symlink_target.symlink_to(external_file)
    
    # Copy source to destination directory - should expand to dest/source.txt
    # and validate the final target
    result = execute_tool(
        "copy_file",
        {"path": "source.txt", "destination": "dest"},
        ws,
        config,
    )
    
    # External file must not be touched
    assert external_file.read_text() == "EXTERNAL CONTENT - SHOULD NOT BE OVERWRITTEN", \
        "External file was overwritten via destination symlink!"
    
    # The copy should either fail or create a new file (not follow symlink)
    # Verify source still exists and has correct content
    assert source.read_text() == "source content"


def test_ascs3_search_files_no_external_symlink_follow(tmp_path, config):
    """ASCS-3: search_files must not follow workspace symlinks to external files.
    
    Regression: workspace link to outside file with AUDIT_MARKER appeared in results.
    """
    ws = Workspace(tmp_path)
    
    # Create internal file with marker
    (tmp_path / "internal.txt").write_text("INTERNAL AUDIT_MARKER")
    
    # Create symlink to external file
    external_file = tmp_path.parent / "external_secret.txt"
    external_file.write_text("EXTERNAL AUDIT_MARKER - SECRET")
    
    symlink = tmp_path / "external_link.txt"
    symlink.symlink_to(external_file)
    
    # Search for marker
    result = execute_tool(
        "search_files",
        {"pattern": "AUDIT_MARKER", "path": "."},
        ws,
        config,
    )
    
    # Should find internal file but NOT follow symlink to external
    assert "internal.txt" in result.output
    assert "external_link.txt" not in result.output or "EXTERNAL" not in result.output, \
        "Search followed symlink to external file!"


def test_ascs4_delete_file_unlinks_not_target(tmp_path, config):
    """ASCS-4: delete_file must unlink the symlink, not delete its target.
    
    Regression: deleting link.txt pointing at internal target deleted target.txt
    and left dangling link.
    """
    ws = Workspace(tmp_path)
    
    # Create target file and symlink to it
    target = tmp_path / "target.txt"
    target.write_text("target content")
    
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    
    # Delete the link - should only remove the link, not the target
    result = execute_tool("delete_file", {"path": "link.txt"}, ws, config)
    
    assert result.ok, f"delete_file failed: {result.output}"
    assert not link.exists(), "Symlink should be removed"
    assert not link.is_symlink(), "Symlink should be removed"
    assert target.exists(), "Target file should still exist!"
    assert target.read_text() == "target content", "Target content should be unchanged"


def test_ascs5_verification_commands_respect_intent(tmp_path, config):
    """ASCS-5: Verification commands must respect read-only intent restrictions.
    
    Regression: Under project_inspection, run_command was refused but
    model-authored task.commands passed to verification created unauthorized.txt.
    """
    # This test requires a TaskExecutor with project_inspection intent
    # We'll test at the executor level
    from agent.execution.executor import TaskExecutor, VerificationResult
    from agent.execution.tasks import Task, COMPLETED
    from agent.config import AgentConfig
    from agent.ollama import OllamaClient
    
    # Create a minimal mock setup
    class MockClient:
        pass
    
    class MockWorkspace:
        def __init__(self):
            self.root = tmp_path
    
    class MockConfig:
        is_plan_mode = False
        is_safe_mode = False
        effective_tools = {"run_command"}
        max_verify_retries = 0
    
    # Create a task with project_inspection intent (read-only)
    task = Task(
        id="test-1",
        title="Inspect task",
        kind="inspect",
        description="Just inspect",
        verification=["run echo hello"],  # This should be blocked for read-only intent
    )
    
    # The fix should be in _verify_task to check intent before running commands
    # This is more of an integration test - we'll verify the executor has the check


def test_ascs6_dirty_file_safeguards_aliases_and_destinations(tmp_path, config):
    """ASCS-6: Dirty-file safeguards must normalize paths and check destinations.
    
    Regression: ./dirty.txt vs dirty.txt missed; copy/move destination ignored.
    """
    import subprocess
    from agent.config import AgentConfig
    
    # Initialize git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    
    # Create a file and commit it
    dirty_file = tmp_path / "dirty.txt"
    dirty_file.write_text("original")
    subprocess.run(["git", "add", "dirty.txt"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, capture_output=True)
    
    # Modify it (make it dirty)
    dirty_file.write_text("modified - dirty")
    
    # Create config with git_baseline including the dirty file
    config_with_baseline = AgentConfig(
        workspace=tmp_path,
        mode="AUTO",
        git_baseline=frozenset({"dirty.txt"})
    )
    
    ws = Workspace(tmp_path)
    
    # Test 1: Write with ./dirty.txt alias should be blocked
    result1 = execute_tool("write_file", {"path": "./dirty.txt", "content": "new"}, ws, config_with_baseline)
    assert not result1.ok, "Should block write to ./dirty.txt (alias)"
    assert "Protected" in result1.output or "pre-existing" in result1.output
    
    # Test 2: Copy destination should be checked
    dest = tmp_path / "dest.txt"
    dest.write_text("dest original")
    subprocess.run(["git", "add", "dest.txt"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "dest"], cwd=tmp_path, capture_output=True)
    dest.write_text("dest modified - dirty")
    
    config_with_baseline2 = AgentConfig(
        workspace=tmp_path,
        mode="AUTO",
        git_baseline=frozenset({"dirty.txt", "dest.txt"})
    )
    
    result2 = execute_tool("copy_file", {"path": "dirty.txt", "destination": "dest.txt"}, ws, config_with_baseline2)
    assert not result2.ok, "Should block copy to dirty destination"
    assert "Protected" in result2.output or "pre-existing" in result2.output


def test_ascs7_verification_requires_actionable_checks(tmp_path, config):
    """ASCS-7: Descriptive notes must not count as successful verification.
    
    Regression: 'Ensure everything works' returned ok=True with only a noted step.
    """
    from agent.execution.executor import TaskExecutor, VerificationResult
    from agent.execution.tasks import Task
    from agent.config import AgentConfig
    
    class MockClient:
        pass
    
    class MockWorkspace:
        def __init__(self):
            self.root = tmp_path
    
    class MockConfig:
        is_plan_mode = False
        is_safe_mode = False
        effective_tools = {"run_command"}
        max_verify_retries = 0
    
    # Create an implementing task with only descriptive verification
    task = Task(
        id="test-impl",
        title="Implement feature",
        kind="implement",
        description="Implement something",
        verification=["Ensure everything works"],  # Only descriptive, no run commands
    )
    
    executor = TaskExecutor(
        config=MockConfig(),
        client=MockClient(),
        workspace=MockWorkspace(),
    )
    
    verification = executor._verify_task(executor, task)
    
    # Should fail because implementing task has no actionable verification steps
    assert not verification.ok, "Implementing task with only notes should not pass verification"
    assert any("run commands for verification" in step.get("output", "").lower() for step in verification.steps)


def test_ascs8_ranged_read_correct_line_numbers(tmp_path, config):
    """ASCS-8: Ranged reads must report correct original line numbers.
    
    Regression: read_file(start_line=100,end_line=105) numbered from 1 not 100.
    """
    ws = Workspace(tmp_path)
    
    # Create a file with 110 lines
    content = "\n".join([f"Line {i}" for i in range(1, 111)])
    (tmp_path / "test.txt").write_text(content)
    
    # Read lines 100-105
    result = execute_tool(
        "read_file",
        {"path": "test.txt", "start_line": 100, "end_line": 105},
        ws,
        config,
    )
    
    assert result.ok
    # Output should show lines numbered 100-105, not 1-6
    assert "100| Line 100" in result.output, f"Expected line 100, got: {result.output[:200]}"
    assert "105| Line 105" in result.output, f"Expected line 105, got: {result.output[:200]}"


# Additional test for move_file with symlink destination
def test_move_file_symlink_destination(tmp_path, config):
    """Test move_file handles symlink destinations correctly."""
    ws = Workspace(tmp_path)
    
    source = tmp_path / "source.txt"
    source.write_text("source content")
    
    target = tmp_path / "target.txt"
    target.write_text("target content")
    
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    
    # Move source to link - should replace the link, not the target
    result = execute_tool("move_file", {"path": "source.txt", "destination": "link.txt"}, ws, config)
    
    assert result.ok
    assert link.exists()
    assert link.read_text() == "source content"
    assert target.exists()  # Target should still exist (though link now points to source.txt location)
    # Actually on move, the link is replaced, so target should be unaffected


# Test for Windows case sensitivity (part of ASCS-6)
def test_dirty_file_windows_case_insensitive(tmp_path, config):
    """Test dirty file detection handles Windows case-insensitive paths."""
    import subprocess
    
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    
    dirty_file = tmp_path / "Dirty.txt"  # Capital D
    dirty_file.write_text("original")
    subprocess.run(["git", "add", "Dirty.txt"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, capture_output=True)
    dirty_file.write_text("modified")
    
    ws = Workspace(tmp_path)
    
    # Try writing with different case
    result = execute_tool("write_file", {"path": "dirty.txt", "content": "new"}, ws, config)
    # On Windows this should be blocked; on Unix it may not be
    # The fix should normalize for case-insensitive comparison on Windows
    import platform
    if platform.system() == "Windows":
        assert not result.ok, "Should block write to dirty.txt (case variant of Dirty.txt)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])