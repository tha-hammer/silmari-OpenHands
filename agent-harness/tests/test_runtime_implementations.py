"""Tests for runtime implementations."""

import pytest
from agent_harness.runtime.local import LocalRuntime


@pytest.mark.asyncio
async def test_local_runtime_setup_and_teardown(tmp_path):
    """Test LocalRuntime setup and teardown."""
    runtime = LocalRuntime(workspace_path=str(tmp_path / "test-workspace"))
    await runtime.setup()

    assert runtime.get_working_directory() == str(tmp_path / "test-workspace")

    await runtime.teardown()


@pytest.mark.asyncio
async def test_local_runtime_execute_cmd_action(tmp_path):
    """Test LocalRuntime can execute command actions."""
    runtime = LocalRuntime(workspace_path=str(tmp_path / "test-workspace"))
    await runtime.setup()

    action = {"type": "cmd", "command": "echo 'hello'"}
    observation = await runtime.execute_action(action)

    assert observation["type"] == "observation"
    assert "hello" in observation["content"]
    assert observation["exit_code"] == 0

    await runtime.teardown()


@pytest.mark.asyncio
async def test_local_runtime_read_and_write_file(tmp_path):
    """Test LocalRuntime can read and write files."""
    runtime = LocalRuntime(workspace_path=str(tmp_path / "test-workspace"))
    await runtime.setup()

    content = "test file content\nline 2"
    await runtime.write_file("test.txt", content)

    read_content = await runtime.read_file("test.txt")
    assert read_content == content

    await runtime.teardown()


@pytest.mark.asyncio
async def test_local_runtime_read_file_with_range(tmp_path):
    """Test LocalRuntime can read file with byte range."""
    runtime = LocalRuntime(workspace_path=str(tmp_path / "test-workspace"))
    await runtime.setup()

    content = "0123456789"
    await runtime.write_file("test.txt", content)

    # Read from start to end
    read_content = await runtime.read_file("test.txt", 0, -1)
    assert read_content == content

    # Read subset
    read_content = await runtime.read_file("test.txt", 2, 7)
    assert read_content == "23456"

    await runtime.teardown()


@pytest.mark.asyncio
async def test_local_runtime_read_nonexistent_file(tmp_path):
    """Test LocalRuntime raises FileNotFoundError for nonexistent files."""
    runtime = LocalRuntime(workspace_path=str(tmp_path / "test-workspace"))
    await runtime.setup()

    with pytest.raises(FileNotFoundError):
        await runtime.read_file("nonexistent.txt")

    await runtime.teardown()

