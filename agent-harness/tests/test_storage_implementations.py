"""Tests for storage implementations."""

import pytest
from agent_harness.storage.local import LocalStorage
from agent_harness.storage.memory import MemoryStorage


@pytest.mark.asyncio
async def test_local_storage_save_and_load_event(tmp_path):
    """Test LocalStorage can save and load events."""
    storage = LocalStorage(base_path=str(tmp_path / "test-harness"))
    session_id = "test-session"

    event = {"id": "event-1", "type": "action", "content": "test"}
    await storage.save_event(session_id, event)

    events = await storage.load_events(session_id)
    assert len(events) == 1
    assert events[0]["id"] == "event-1"
    assert events[0]["type"] == "action"


@pytest.mark.asyncio
async def test_local_storage_save_and_load_file(tmp_path):
    """Test LocalStorage can save and load files."""
    storage = LocalStorage(base_path=str(tmp_path / "test-harness"))
    
    content = b"test file content"
    await storage.save_file("test/file.txt", content)
    
    loaded_content = await storage.load_file("test/file.txt")
    assert loaded_content == content


@pytest.mark.asyncio
async def test_local_storage_list_files(tmp_path):
    """Test LocalStorage can list files."""
    storage = LocalStorage(base_path=str(tmp_path / "test-harness"))
    
    await storage.save_file("dir1/file1.txt", b"content1")
    await storage.save_file("dir1/file2.txt", b"content2")
    await storage.save_file("dir2/file3.txt", b"content3")
    
    files = await storage.list_files("dir1")
    assert len(files) == 2
    assert "dir1/file1.txt" in files
    assert "dir1/file2.txt" in files


@pytest.mark.asyncio
async def test_memory_storage_save_and_load_event():
    """Test MemoryStorage can save and load events."""
    storage = MemoryStorage()
    session_id = "test-session"

    event = {"id": "event-1", "type": "action", "content": "test"}
    await storage.save_event(session_id, event)

    events = await storage.load_events(session_id)
    assert len(events) == 1
    assert events[0]["id"] == "event-1"
    assert events[0]["type"] == "action"


@pytest.mark.asyncio
async def test_memory_storage_save_and_load_file():
    """Test MemoryStorage can save and load files."""
    storage = MemoryStorage()
    
    content = b"test file content"
    await storage.save_file("test/file.txt", content)
    
    loaded_content = await storage.load_file("test/file.txt")
    assert loaded_content == content


@pytest.mark.asyncio
async def test_memory_storage_list_files():
    """Test MemoryStorage can list files."""
    storage = MemoryStorage()
    
    await storage.save_file("dir1/file1.txt", b"content1")
    await storage.save_file("dir1/file2.txt", b"content2")
    await storage.save_file("dir2/file3.txt", b"content3")
    
    files = await storage.list_files("dir1")
    assert len(files) == 2
    assert "dir1/file1.txt" in files
    assert "dir1/file2.txt" in files


@pytest.mark.asyncio
async def test_memory_storage_load_nonexistent_file():
    """Test MemoryStorage raises FileNotFoundError for nonexistent files."""
    storage = MemoryStorage()
    
    with pytest.raises(FileNotFoundError):
        await storage.load_file("nonexistent.txt")

