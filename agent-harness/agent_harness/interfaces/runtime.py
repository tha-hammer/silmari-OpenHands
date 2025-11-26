"""Runtime interface for action execution."""

from abc import ABC, abstractmethod
from typing import Any


class RuntimeInterface(ABC):
    """Abstract interface for runtime execution."""

    @abstractmethod
    async def execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute an action and return observation.

        Args:
            action: Action dictionary with type and parameters

        Returns:
            Observation dictionary with results
        """
        pass

    @abstractmethod
    def get_working_directory(self) -> str:
        """Get current working directory.

        Returns:
            Absolute path to working directory
        """
        pass

    @abstractmethod
    async def setup(self) -> None:
        """Initialize runtime environment."""
        pass

    @abstractmethod
    async def teardown(self) -> None:
        """Clean up runtime environment."""
        pass

    @abstractmethod
    async def read_file(self, path: str, start: int = 0, end: int = -1) -> str:
        """Read file content.

        Args:
            path: File path relative to working directory
            start: Start byte offset (default: 0)
            end: End byte offset (-1 for end of file)

        Returns:
            File content as string
        """
        pass

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write file content.

        Args:
            path: File path relative to working directory
            content: Content to write as string
        """
        pass

