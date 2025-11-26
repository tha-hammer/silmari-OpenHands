"""Main AgentHarness class for running agents."""

import uuid
from typing import Any, Callable, Optional

from agent_harness.config import HarnessConfig
from agent_harness.core.agent import Agent
from agent_harness.core.controller import AgentController, AgentState
from agent_harness.core.state import State
from agent_harness.events.stream import EventStream
from agent_harness.interfaces.runtime import RuntimeInterface
from agent_harness.interfaces.storage import StorageInterface
from agent_harness.runtime.local import LocalRuntime
from agent_harness.storage.local import LocalStorage
from agent_harness.utils.logging import setup_logger

logger = setup_logger(__name__)


class AgentHarness:
    """Main harness class for running agents.

    This is the primary entry point for using the agent harness library.
    """

    def __init__(
        self,
        config: HarnessConfig,
        agent: Optional[Agent] = None,
        storage: Optional[StorageInterface] = None,
        runtime: Optional[RuntimeInterface] = None,
        status_callback: Optional[Callable[[str, str, str], None]] = None,
    ):
        """Initialize agent harness.

        Args:
            config: Harness configuration
            agent: Optional agent instance (will be created from config if not provided)
            storage: Optional storage implementation (defaults to LocalStorage)
            runtime: Optional runtime implementation (defaults to LocalRuntime)
            status_callback: Optional callback for status updates
        """
        self.config = config
        self.status_callback = status_callback

        # Create storage if not provided
        if storage is None:
            storage = LocalStorage(base_path=config.storage_path)
        self.storage = storage

        # Create runtime if not provided
        if runtime is None:
            runtime = LocalRuntime(workspace_path=config.workspace_path)
        self.runtime = runtime

        # Create agent if not provided
        if agent is None:
            agent_class = Agent.get_agent_class(config.agent_name)
            agent = agent_class(config)
        self.agent = agent

        # Create session ID
        self.session_id = str(uuid.uuid4())

        # Initialize components
        self.event_stream: Optional[EventStream] = None
        self.controller: Optional[AgentController] = None
        self.state: Optional[State] = None

    async def initialize(self) -> None:
        """Initialize harness components."""
        # Create event stream
        self.event_stream = EventStream(
            session_id=self.session_id,
            storage=self.storage,
        )
        await self.event_stream.initialize()

        # Create or restore state
        try:
            self.state = await State.restore_from_session(
                self.session_id, self.storage
            )
        except Exception:
            # Create new state if restore fails
            self.state = State(session_id=self.session_id)

        # Set max iterations from config
        self.state.max_iterations = self.config.max_iterations

        # Create controller
        self.controller = AgentController(
            agent=self.agent,
            event_stream=self.event_stream,
            state=self.state,
            storage=self.storage,
            runtime=self.runtime,
            max_iterations=self.config.max_iterations,
            status_callback=self.status_callback,
        )

        # Setup runtime
        await self.runtime.setup()

        logger.info(f"AgentHarness initialized with session {self.session_id}")

    async def run(self, task: str) -> dict[str, Any]:
        """Run agent with a task.

        Args:
            task: Task description

        Returns:
            Final outputs from agent
        """
        if self.controller is None:
            await self.initialize()

        if self.controller is None:
            raise RuntimeError("Failed to initialize controller")

        logger.info(f"Running task: {task}")
        result = await self.controller.run(task)
        logger.info(f"Task completed with outputs: {result}")
        return result

    async def close(self) -> None:
        """Close harness and clean up resources."""
        if self.controller:
            await self.controller.close()

        if self.event_stream:
            await self.event_stream.close()

        if self.runtime:
            await self.runtime.teardown()

        logger.info("AgentHarness closed")

    async def __aenter__(self) -> "AgentHarness":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

