"""Action types for agent harness."""

from dataclasses import dataclass, field
from typing import Any

from agent_harness.events.event import Event, EventSource


@dataclass
class Action(Event):
    """Base action class."""

    runnable: bool = False


@dataclass
class MessageAction(Action):
    """Action representing a message."""

    content: str = ""
    wait_for_response: bool = False

    @property
    def message(self) -> str:
        """Get message content."""
        return self.content

    def __str__(self) -> str:
        return f"MessageAction: {self.content}"


@dataclass
class SystemMessageAction(Action):
    """System message action with tools."""

    content: str = ""
    tools: list[Any] | None = None

    @property
    def message(self) -> str:
        """Get message content."""
        return self.content

    def __str__(self) -> str:
        return f"SystemMessageAction: {self.content[:50]}..."


@dataclass
class AgentFinishAction(Action):
    """Action when agent finishes task."""

    final_thought: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)
    thought: str = ""

    @property
    def message(self) -> str:
        """Get finish message."""
        return self.thought or self.final_thought or "Task completed"

    def __str__(self) -> str:
        return f"AgentFinishAction: {self.message}"


@dataclass
class AgentRejectAction(Action):
    """Action when agent rejects task."""

    outputs: dict[str, Any] = field(default_factory=dict)
    thought: str = ""

    @property
    def message(self) -> str:
        """Get reject message."""
        reason = self.outputs.get("reason", "")
        return f"Task rejected: {reason}" if reason else "Task rejected"

    def __str__(self) -> str:
        return f"AgentRejectAction: {self.message}"


@dataclass
class NullAction(Action):
    """Null action (no-op)."""

    def __str__(self) -> str:
        return "NullAction"

