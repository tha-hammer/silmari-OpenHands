"""Event stream for managing and persisting events."""

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Optional

from agent_harness.events.event import Event, EventSource
from agent_harness.interfaces.storage import StorageInterface
from agent_harness.utils.logging import setup_logger

logger = setup_logger()


class EventStream:
    """Event stream for managing and persisting events."""

    def __init__(
        self,
        session_id: str,
        storage: StorageInterface,
    ):
        """Initialize event stream.

        Args:
            session_id: Unique session identifier
            storage: Storage interface for persistence
        """
        self.session_id = session_id
        self.storage = storage
        self._cur_id = 0
        self._subscribers: dict[str, Callable[[Event], None]] = {}

    async def _load_cur_id(self) -> int:
        """Load current event ID from storage."""
        events = await self.storage.load_events(self.session_id)
        if not events:
            return 0
        max_id = max((e.get("id", 0) for e in events), default=0)
        return max_id + 1

    async def initialize(self) -> None:
        """Initialize event stream by loading current ID."""
        self._cur_id = await self._load_cur_id()

    def subscribe(self, callback_id: str, callback: Callable[[Event], None]) -> None:
        """Subscribe to events.

        Args:
            callback_id: Unique identifier for the callback
            callback: Callback function to call when events are added
        """
        if callback_id in self._subscribers:
            raise ValueError(f"Callback ID already exists: {callback_id}")
        self._subscribers[callback_id] = callback

    def unsubscribe(self, callback_id: str) -> None:
        """Unsubscribe from events.

        Args:
            callback_id: Callback identifier to remove
        """
        if callback_id in self._subscribers:
            del self._subscribers[callback_id]

    async def add_event(self, event: Event, source: EventSource) -> None:
        """Add event to stream.

        Args:
            event: Event to add
            source: Source of the event
        """
        if event.id != Event.INVALID_ID:
            raise ValueError(
                f"Event already has an ID: {event.id}. "
                "It was probably added back to the EventStream from inside a handler."
            )

        event._timestamp = datetime.now().isoformat()
        event._source = source
        event._id = self._cur_id
        self._cur_id += 1

        # Save to storage
        event_dict = event.to_dict()
        await self.storage.save_event(self.session_id, event_dict)

        # Notify subscribers
        for callback in self._subscribers.values():
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber callback: {e}")

    async def get_event(self, event_id: int) -> Optional[Event]:
        """Get event by ID.

        Args:
            event_id: Event ID to retrieve

        Returns:
            Event if found, None otherwise
        """
        events = await self.storage.load_events(self.session_id)
        for event_dict in events:
            if event_dict.get("id") == event_id:
                return Event.from_dict(event_dict)
        return None

    async def get_events(
        self, start_id: int = 0, end_id: Optional[int] = None
    ) -> list[Event]:
        """Get events in range.

        Args:
            start_id: Starting event ID (inclusive)
            end_id: Ending event ID (exclusive, None for all)

        Returns:
            List of events
        """
        events = await self.storage.load_events(self.session_id)
        result = []
        for event_dict in events:
            event_id = event_dict.get("id", Event.INVALID_ID)
            if event_id >= start_id:
                if end_id is None or event_id < end_id:
                    result.append(Event.from_dict(event_dict))
        return sorted(result, key=lambda e: e.id)

    @property
    def cur_id(self) -> int:
        """Get current event ID."""
        return self._cur_id

    async def close(self) -> None:
        """Close event stream and clean up."""
        self._subscribers.clear()

