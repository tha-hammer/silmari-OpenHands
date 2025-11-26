"""Tests for event system."""

import pytest
from agent_harness.events import Event, EventSource, EventStream
from agent_harness.storage.memory import MemoryStorage


@pytest.mark.asyncio
async def test_event_creation():
    """Test creating events."""
    event = Event()
    assert event.id == Event.INVALID_ID
    assert event.source is None
    assert event.timestamp is None


@pytest.mark.asyncio
async def test_event_to_dict():
    """Test event serialization."""
    event = Event()
    event._id = 1
    event._source = EventSource.AGENT
    event._message = "test message"
    
    event_dict = event.to_dict()
    assert event_dict["id"] == 1
    assert event_dict["source"] == "agent"
    assert event_dict["message"] == "test message"


@pytest.mark.asyncio
async def test_event_from_dict():
    """Test event deserialization."""
    event_dict = {
        "id": 1,
        "source": "agent",
        "message": "test message",
        "type": "Event",
    }
    event = Event.from_dict(event_dict)
    assert event.id == 1
    assert event.source == EventSource.AGENT
    assert event.message == "test message"


@pytest.mark.asyncio
async def test_event_stream_add_event():
    """Test adding events to stream."""
    storage = MemoryStorage()
    stream = EventStream("test-session", storage)
    await stream.initialize()

    event = Event()
    await stream.add_event(event, EventSource.AGENT)

    assert stream.cur_id == 1
    assert event.id == 0


@pytest.mark.asyncio
async def test_event_stream_get_event():
    """Test retrieving events from stream."""
    storage = MemoryStorage()
    stream = EventStream("test-session", storage)
    await stream.initialize()

    event = Event()
    await stream.add_event(event, EventSource.AGENT)

    retrieved = await stream.get_event(0)
    assert retrieved is not None
    assert retrieved.id == 0


@pytest.mark.asyncio
async def test_event_stream_subscribe():
    """Test event subscription."""
    storage = MemoryStorage()
    stream = EventStream("test-session", storage)
    await stream.initialize()

    events_received = []

    def callback(event: Event) -> None:
        events_received.append(event)

    stream.subscribe("test-callback", callback)

    event = Event()
    await stream.add_event(event, EventSource.AGENT)

    assert len(events_received) == 1
    assert events_received[0].id == 0

