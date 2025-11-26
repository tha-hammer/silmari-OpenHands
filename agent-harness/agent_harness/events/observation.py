"""Observation types for agent harness."""

from dataclasses import dataclass
from typing import Any

from agent_harness.events.event import Event


@dataclass
class Observation(Event):
    """Base observation class."""

    content: str = ""
    cause: int | None = None  # ID of action that caused this observation

    def __str__(self) -> str:
        return f"Observation: {self.content[:100]}..."


@dataclass
class CmdOutputObservation(Observation):
    """Observation from command execution."""

    command: str = ""
    exit_code: int = 0
    error: str | None = None

    def __str__(self) -> str:
        if self.error:
            return f"CmdOutputObservation (error): {self.error}"
        return f"CmdOutputObservation (exit={self.exit_code}): {self.content[:100]}..."


@dataclass
class ErrorObservation(Observation):
    """Observation representing an error."""

    error_id: str | None = None

    def __str__(self) -> str:
        return f"ErrorObservation: {self.content}"


@dataclass
class NullObservation(Observation):
    """Null observation (no-op)."""

    def __str__(self) -> str:
        return "NullObservation"

