"""In-memory storage implementation for testing."""

from typing import Any

from agent_harness.interfaces.storage import StorageInterface


class MemoryStorage(StorageInterface):
    """In-memory storage implementation."""

    def __init__(self):
        """Initialize in-memory storage."""
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._files: dict[str, bytes] = {}

    async def save_event(self, session_id: str, event: dict[str, Any]) -> None:
        """Save event to memory."""
        if session_id not in self._events:
            self._events[session_id] = []
        self._events[session_id].append(event)

    async def load_events(self, session_id: str) -> list[dict[str, Any]]:
        """Load events from memory."""
        return self._events.get(session_id, []).copy()

    async def save_file(self, path: str, content: bytes) -> None:
        """Save file to memory."""
        self._files[path] = content

    async def load_file(self, path: str) -> bytes:
        """Load file from memory."""
        if path not in self._files:
            raise FileNotFoundError(f"File not found: {path}")
        return self._files[path]

    async def list_files(self, prefix: str) -> list[str]:
        """List files with given prefix."""
        files = []
        for file_path in self._files.keys():
            if file_path.startswith(prefix):
                files.append(file_path)
        return sorted(files)

