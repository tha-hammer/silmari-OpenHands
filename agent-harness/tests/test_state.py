"""Tests for state management."""

import pytest
from agent_harness.core.state import State
from agent_harness.events.event import Event
from agent_harness.storage.memory import MemoryStorage


@pytest.mark.asyncio
async def test_state_creation():
    """Test creating state."""
    state = State(session_id="test-session")
    assert state.session_id == "test-session"
    assert state.iteration == 0
    assert len(state.history) == 0


@pytest.mark.asyncio
async def test_state_save_and_restore():
    """Test saving and restoring state."""
    storage = MemoryStorage()
    state = State(
        session_id="test-session",
        iteration=5,
        inputs={"key": "value"},
        outputs={"result": "success"},
    )

    await state.save_to_session("test-session", storage)

    restored = await State.restore_from_session("test-session", storage)
    assert restored.session_id == "test-session"
    assert restored.iteration == 5
    assert restored.inputs == {"key": "value"}
    assert restored.outputs == {"result": "success"}


@pytest.mark.asyncio
async def test_state_restore_nonexistent():
    """Test restoring non-existent state."""
    storage = MemoryStorage()
    restored = await State.restore_from_session("nonexistent", storage)
    assert restored.session_id == "nonexistent"
    assert restored.iteration == 0

