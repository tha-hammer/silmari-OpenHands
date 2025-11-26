"""Local file system storage implementation."""

import json
import os
from pathlib import Path
from typing import Any

from agent_harness.interfaces.storage import StorageInterface


class LocalStorage(StorageInterface):
    """Local file system storage implementation."""

    def __init__(self, base_path: str = "~/.agent-harness"):
        """Initialize local storage.

        Args:
            base_path: Base directory for storage (default: ~/.agent-harness)
        """
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save_event(self, session_id: str, event: dict[str, Any]) -> None:
        """Save event to local file system."""
        events_dir = self.base_path / "sessions" / session_id / "events"
        events_dir.mkdir(parents=True, exist_ok=True)

        event_id = event.get("id", "event")
        event_file = events_dir / f"{event_id}.json"
        with open(event_file, 'w') as f:
            json.dump(event, f, indent=2)

    async def load_events(self, session_id: str) -> list[dict[str, Any]]:
        """Load events from local file system."""
        events_dir = self.base_path / "sessions" / session_id / "events"
        if not events_dir.exists():
            return []

        events = []
        for event_file in sorted(events_dir.glob("*.json")):
            with open(event_file, 'r') as f:
                events.append(json.load(f))

        return events

    async def save_file(self, path: str, content: bytes) -> None:
        """Save file to local file system."""
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'wb') as f:
            f.write(content)

    async def load_file(self, path: str) -> bytes:
        """Load file from local file system."""
        full_path = self.base_path / path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(full_path, 'rb') as f:
            return f.read()

    async def list_files(self, prefix: str) -> list[str]:
        """List files with given prefix."""
        prefix_path = self.base_path / prefix
        if not prefix_path.exists():
            return []

        files = []
        for item in prefix_path.rglob("*"):
            if item.is_file():
                # Return relative path from base_path
                relative_path = item.relative_to(self.base_path)
                files.append(str(relative_path))

        return sorted(files)

