"""Storage interface for event and file persistence."""

from abc import ABC, abstractmethod
from typing import Any


class StorageInterface(ABC):
    """Abstract interface for event and file storage."""

    @abstractmethod
    async def save_event(self, session_id: str, event: dict[str, Any]) -> None:
        """Save event to storage.

        Args:
            session_id: Unique identifier for the session
            event: Event data dictionary to save
        """
        pass

    @abstractmethod
    async def load_events(self, session_id: str) -> list[dict[str, Any]]:
        """Load events for session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            List of event dictionaries
        """
        pass

    @abstractmethod
    async def save_file(self, path: str, content: bytes) -> None:
        """Save file to storage.

        Args:
            path: File path relative to storage root
            content: File content as bytes
        """
        pass

    @abstractmethod
    async def load_file(self, path: str) -> bytes:
        """Load file from storage.

        Args:
            path: File path relative to storage root

        Returns:
            File content as bytes
        """
        pass

    @abstractmethod
    async def list_files(self, prefix: str) -> list[str]:
        """List files with given prefix.

        Args:
            prefix: Path prefix to filter files

        Returns:
            List of file paths matching the prefix
        """
        pass

