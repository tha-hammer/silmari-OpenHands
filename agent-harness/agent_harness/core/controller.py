"""Agent controller for managing agent execution."""

import asyncio
from typing import Any, Callable, Optional

from agent_harness.core.agent import Agent
from agent_harness.core.state import State
from agent_harness.events.action import (
    Action,
    AgentFinishAction,
    AgentRejectAction,
    MessageAction,
    NullAction,
)
from agent_harness.events.event import Event, EventSource
from agent_harness.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    NullObservation,
    Observation,
)
from agent_harness.events.stream import EventStream
from agent_harness.interfaces.runtime import RuntimeInterface
from agent_harness.interfaces.storage import StorageInterface
from agent_harness.utils.logging import setup_logger

logger = setup_logger(__name__)


class AgentState:
    """Agent execution states."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"
    STOPPED = "stopped"
    AWAITING_USER_INPUT = "awaiting_user_input"


class AgentController:
    """Controller for managing agent execution."""

    def __init__(
        self,
        agent: Agent,
        event_stream: EventStream,
        state: State,
        storage: StorageInterface,
        runtime: RuntimeInterface,
        max_iterations: int = 100,
        status_callback: Optional[Callable[[str, str, str], None]] = None,
    ):
        """Initialize agent controller.

        Args:
            agent: Agent instance to control
            event_stream: Event stream for publishing events
            state: Initial state
            storage: Storage interface
            runtime: Runtime interface for executing actions
            max_iterations: Maximum number of iterations
            status_callback: Optional callback for status updates
        """
        self.agent = agent
        self.event_stream = event_stream
        self.state = state
        self.storage = storage
        self.runtime = runtime
        self.max_iterations = max_iterations
        self.status_callback = status_callback
        self._closed = False
        self._pending_action: Optional[Action] = None

        # Subscribe to event stream
        self.event_stream.subscribe("agent_controller", self.on_event)

        # Set initial state
        self._agent_state = AgentState.INITIALIZING

    @property
    def agent_state(self) -> str:
        """Get current agent state."""
        return self._agent_state

    def on_event(self, event: Event) -> None:
        """Handle event from event stream.

        Args:
            event: Event to handle
        """
        if self._closed:
            return

        # Add to state history
        self.state.history.append(event)

        # Handle based on event type
        if isinstance(event, Action):
            asyncio.create_task(self._handle_action(event))
        elif isinstance(event, Observation):
            asyncio.create_task(self._handle_observation(event))

        # Check if we should step
        if self._should_step(event):
            asyncio.create_task(self._step())

    async def _handle_action(self, action: Action) -> None:
        """Handle action from event stream.

        Args:
            action: Action to handle
        """
        if isinstance(action, AgentFinishAction):
            self.state.outputs = action.outputs
            await self._set_agent_state(AgentState.FINISHED)
        elif isinstance(action, AgentRejectAction):
            self.state.outputs = action.outputs
            await self._set_agent_state(AgentState.ERROR)
        elif isinstance(action, MessageAction) and action.source == EventSource.USER:
            logger.info(f"User message: {action.content}")
            if self._agent_state != AgentState.RUNNING:
                await self._set_agent_state(AgentState.RUNNING)

    async def _handle_observation(self, observation: Observation) -> None:
        """Handle observation from event stream.

        Args:
            observation: Observation to handle
        """
        logger.debug(f"Observation: {observation.content[:100]}...")

        # Clear pending action if this observation is for it
        if (
            self._pending_action
            and observation.cause
            and self._pending_action.id == observation.cause
        ):
            self._pending_action = None
            if self._agent_state == AgentState.AWAITING_USER_INPUT:
                await self._set_agent_state(AgentState.RUNNING)

    def _should_step(self, event: Event) -> bool:
        """Check if agent should step based on event.

        Args:
            event: Event to check

        Returns:
            True if agent should step
        """
        if self._agent_state != AgentState.RUNNING:
            return False

        if self._pending_action:
            return False

        if isinstance(event, MessageAction) and event.source == EventSource.USER:
            return True

        if isinstance(event, Observation) and not isinstance(event, NullObservation):
            return True

        return False

    async def _step(self) -> None:
        """Execute one step of the agent."""
        if self._agent_state != AgentState.RUNNING:
            logger.debug(
                f"Not stepping: agent state is {self._agent_state} (not RUNNING)"
            )
            return

        if self._pending_action:
            logger.debug("Not stepping: pending action exists")
            return

        # Check iteration limit
        if self.state.iteration >= self.max_iterations:
            await self._set_agent_state(AgentState.ERROR)
            await self.event_stream.add_event(
                ErrorObservation(
                    content=f"Maximum iterations ({self.max_iterations}) reached"
                ),
                EventSource.ENVIRONMENT,
            )
            return

        # Increment iteration
        self.state.iteration += 1
        logger.debug(f"Step {self.state.iteration}/{self.max_iterations}")

        try:
            # Get action from agent
            action = await self.agent.step(self.state)
            if action is None:
                action = NullAction()

            action._source = EventSource.AGENT  # type: ignore[attr-defined]

            # Add action to event stream first (so it gets an ID)
            if not isinstance(action, NullAction):
                await self.event_stream.add_event(action, EventSource.AGENT)

            # Handle runnable actions
            if action.runnable:
                self._pending_action = action
                await self._execute_action(action)

        except Exception as e:
            logger.error(f"Error in agent step: {e}", exc_info=True)
            await self._set_agent_state(AgentState.ERROR)
            await self.event_stream.add_event(
                ErrorObservation(content=f"Agent error: {str(e)}"),
                EventSource.ENVIRONMENT,
            )

    async def _execute_action(self, action: Action) -> None:
        """Execute a runnable action.

        Args:
            action: Action to execute
        """
        logger.debug(f"Executing action: {type(action).__name__}")

        try:
            # Convert action to runtime format
            action_dict = self._action_to_dict(action)

            # Execute via runtime
            result = await self.runtime.execute_action(action_dict)

            # Create observation
            observation = self._result_to_observation(result, action.id)
            observation.cause = action.id

            # Add to event stream
            await self.event_stream.add_event(observation, EventSource.ENVIRONMENT)

        except Exception as e:
            logger.error(f"Error executing action: {e}", exc_info=True)
            error_obs = ErrorObservation(
                content=f"Action execution error: {str(e)}",
                cause=action.id,
            )
            await self.event_stream.add_event(error_obs, EventSource.ENVIRONMENT)

    def _action_to_dict(self, action: Action) -> dict[str, Any]:
        """Convert action to dictionary for runtime.

        Args:
            action: Action to convert

        Returns:
            Dictionary representation
        """
        # Simple conversion - can be extended for specific action types
        return {
            "type": type(action).__name__.lower().replace("action", ""),
            "content": getattr(action, "content", ""),
            "command": getattr(action, "command", ""),
            "path": getattr(action, "path", ""),
        }

    def _result_to_observation(
        self, result: dict[str, Any], cause_id: int
    ) -> Observation:
        """Convert runtime result to observation.

        Args:
            result: Result from runtime
            cause_id: ID of action that caused this observation

        Returns:
            Observation instance
        """
        obs_type = result.get("type", "observation")
        content = result.get("content", "")
        error = result.get("error")

        if obs_type == "cmd_output" or "command" in result:
            return CmdOutputObservation(
                content=content,
                command=result.get("command", ""),
                exit_code=result.get("exit_code", 0),
                error=error,
                cause=cause_id,
            )

        if error:
            return ErrorObservation(content=error or content, cause=cause_id)

        return Observation(content=content, cause=cause_id)

    async def _set_agent_state(self, new_state: str) -> None:
        """Set agent state.

        Args:
            new_state: New state to set
        """
        if new_state == self._agent_state:
            return

        old_state = self._agent_state
        self._agent_state = new_state

        logger.info(f"Agent state: {old_state} -> {new_state}")

        # Save state
        await self.state.save_to_session(self.state.session_id, self.storage)

        # Call status callback if provided
        if self.status_callback:
            self.status_callback("state_change", new_state, "")

    async def run(self, task: str) -> dict[str, Any]:
        """Run agent with a task.

        Args:
            task: Task description

        Returns:
            Final outputs from agent
        """
        # Initialize
        await self.event_stream.initialize()
        await self._set_agent_state(AgentState.RUNNING)

        # Add user message
        user_message = MessageAction(content=task)
        await self.event_stream.add_event(user_message, EventSource.USER)

        # Wait for completion
        while self._agent_state not in (AgentState.FINISHED, AgentState.ERROR, AgentState.STOPPED):
            await asyncio.sleep(0.1)

        return self.state.outputs

    async def close(self) -> None:
        """Close controller and clean up."""
        if self._closed:
            return

        self._closed = True
        await self._set_agent_state(AgentState.STOPPED)

        # Unsubscribe from event stream
        self.event_stream.unsubscribe("agent_controller")

        # Save final state
        await self.state.save_to_session(self.state.session_id, self.storage)

        logger.info("Agent controller closed")

