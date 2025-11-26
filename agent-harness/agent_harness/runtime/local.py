"""Local process runtime implementation."""

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any

from agent_harness.interfaces.runtime import RuntimeInterface


class LocalRuntime(RuntimeInterface):
    """Local process runtime implementation."""

    def __init__(self, workspace_path: str = "./workspace"):
        """Initialize local runtime.

        Args:
            workspace_path: Path to workspace directory
        """
        self.workspace_path = Path(workspace_path).resolve()
        self.workspace_path.mkdir(parents=True, exist_ok=True)

    async def setup(self) -> None:
        """Initialize runtime environment."""
        # Ensure workspace exists
        self.workspace_path.mkdir(parents=True, exist_ok=True)

    async def teardown(self) -> None:
        """Clean up runtime environment."""
        # Nothing to clean up for local runtime
        pass

    def get_working_directory(self) -> str:
        """Get current working directory."""
        return str(self.workspace_path)

    async def execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute action in local process."""
        action_type = action.get("type")

        if action_type == "cmd":
            command = action.get("command", "")
            result = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_path
            )
            stdout, stderr = await result.communicate()

            return {
                "type": "observation",
                "content": stdout.decode('utf-8', errors='replace'),
                "error": stderr.decode('utf-8', errors='replace') if result.returncode != 0 else None,
                "exit_code": result.returncode
            }

        # Handle other action types...
        return {"type": "observation", "content": ""}

    async def read_file(self, path: str, start: int = 0, end: int = -1) -> str:
        """Read file content."""
        file_path = self.workspace_path / path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            if end == -1:
                return content[start:]
            return content[start:end]

    async def write_file(self, path: str, content: str) -> None:
        """Write file content."""
        file_path = self.workspace_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

