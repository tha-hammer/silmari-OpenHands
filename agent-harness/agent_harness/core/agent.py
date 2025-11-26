"""Agent base class for agent harness."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from agent_harness.config import HarnessConfig
from agent_harness.events.event import EventSource
from agent_harness.utils.logging import setup_logger

logger = setup_logger()


class Agent(ABC):
    """Abstract base class for agents.

    This is a simplified version for the standalone harness.
    """

    _registry: dict[str, type["Agent"]] = {}

    def __init__(self, config: HarnessConfig):
        """Initialize agent.

        Args:
            config: Harness configuration
        """
        self.config = config
        self._complete = False

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.__class__.__name__

    @property
    def complete(self) -> bool:
        """Check if agent execution is complete."""
        return self._complete

    @complete.setter
    def complete(self, value: bool) -> None:
        """Set completion status."""
        self._complete = value

    @abstractmethod
    async def step(self, event: Any) -> Any:
        """Execute one step of the agent.

        Args:
            event: Input event

        Returns:
            Output event or observation
        """
        pass

    @classmethod
    def register(cls, name: Optional[str] = None) -> type["Agent"]:
        """Register agent class.

        Args:
            name: Optional name for the agent (defaults to class name)

        Returns:
            The agent class (for use as decorator)
        """
        agent_name = name or cls.__name__
        if agent_name in Agent._registry:
            raise ValueError(f"Agent {agent_name} already registered")
        Agent._registry[agent_name] = cls
        return cls

    @classmethod
    def get_agent_class(cls, name: str) -> type["Agent"]:
        """Get agent class by name.

        Args:
            name: Agent name

        Returns:
            Agent class

        Raises:
            ValueError: If agent not found
        """
        if name not in Agent._registry:
            raise ValueError(f"Agent {name} not registered")
        return Agent._registry[name]

