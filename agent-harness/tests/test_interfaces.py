"""Tests for interface contracts."""

import pytest
from agent_harness.interfaces import StorageInterface, RuntimeInterface


def test_storage_interface_contract():
    """Test that StorageInterface defines required methods."""
    assert hasattr(StorageInterface, 'save_event')
    assert hasattr(StorageInterface, 'load_events')
    assert hasattr(StorageInterface, 'save_file')
    assert hasattr(StorageInterface, 'load_file')
    assert hasattr(StorageInterface, 'list_files')


def test_runtime_interface_contract():
    """Test that RuntimeInterface defines required methods."""
    assert hasattr(RuntimeInterface, 'execute_action')
    assert hasattr(RuntimeInterface, 'get_working_directory')
    assert hasattr(RuntimeInterface, 'setup')
    assert hasattr(RuntimeInterface, 'teardown')
    assert hasattr(RuntimeInterface, 'read_file')
    assert hasattr(RuntimeInterface, 'write_file')


@pytest.mark.asyncio
async def test_storage_interface_cannot_instantiate():
    """Test that abstract interface cannot be instantiated."""
    with pytest.raises(TypeError):
        StorageInterface()  # Should fail - abstract class


@pytest.mark.asyncio
async def test_runtime_interface_cannot_instantiate():
    """Test that abstract interface cannot be instantiated."""
    with pytest.raises(TypeError):
        RuntimeInterface()  # Should fail - abstract class

