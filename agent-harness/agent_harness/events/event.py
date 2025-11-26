"""Event base class and types."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class EventSource(str, Enum):
    """Source of an event."""

    AGENT = 'agent'
    USER = 'user'
    ENVIRONMENT = 'environment'


@dataclass
class Event:
    """Base event class for agent harness."""

    INVALID_ID = -1

    _id: int = INVALID_ID
    _timestamp: Optional[str] = None
    _source: Optional[EventSource] = None
    _message: Optional[str] = None

    @property
    def id(self) -> int:
        """Get event ID."""
        return self._id if self._id is not None else Event.INVALID_ID

    @property
    def timestamp(self) -> Optional[str]:
        """Get event timestamp."""
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: datetime) -> None:
        """Set event timestamp."""
        if isinstance(value, datetime):
            self._timestamp = value.isoformat()

    @property
    def source(self) -> Optional[EventSource]:
        """Get event source."""
        return self._source

    @property
    def message(self) -> Optional[str]:
        """Get event message."""
        return self._message

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self._id,
            "timestamp": self._timestamp,
            "source": self._source.value if self._source else None,
            "message": self._message,
            "type": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        event = cls()
        event._id = data.get("id", Event.INVALID_ID)
        event._timestamp = data.get("timestamp")
        if data.get("source"):
            event._source = EventSource(data["source"])
        event._message = data.get("message")
        return event

