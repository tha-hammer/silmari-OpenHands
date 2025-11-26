"""Tests for AgentController."""

import pytest
from agent_harness.core.agent import Agent
from agent_harness.core.controller import AgentController, AgentState
from agent_harness.core.state import State
from agent_harness.events.action import AgentFinishAction, MessageAction, NullAction
from agent_harness.events.event import EventSource
from agent_harness.events.observation import NullObservation, Observation
from agent_harness.events.stream import EventStream
from agent_harness.storage.memory import MemoryStorage
from agent_harness.runtime.local import LocalRuntime


class TestAgent(Agent):
    """Test agent implementation."""

    def __init__(self, config):
        super().__init__(config)
        self.step_count = 0

    async def step(self, state):
        """Simple test step."""
        self.step_count += 1
        if self.step_count >= 3:
            return AgentFinishAction(outputs={"result": "done"})
        return NullAction()


@pytest.mark.asyncio
async def test_controller_initialization():
    """Test controller can be initialized."""
    from agent_harness.config import HarnessConfig, LLMConfig

    config = HarnessConfig(llm=LLMConfig(model="test"))
    agent = TestAgent(config)
    storage = MemoryStorage()
    runtime = LocalRuntime(workspace_path="/tmp/test-workspace")
    event_stream = EventStream(session_id="test-session", storage=storage)
    await event_stream.initialize()

    state = State(session_id="test-session")

    controller = AgentController(
        agent=agent,
        event_stream=event_stream,
        state=state,
        storage=storage,
        runtime=runtime,
        max_iterations=10,
    )

    assert controller.agent == agent
    assert controller.agent_state == AgentState.INITIALIZING
    await controller.close()


@pytest.mark.asyncio
async def test_controller_handles_user_message():
    """Test controller handles user messages."""
    from agent_harness.config import HarnessConfig, LLMConfig

    config = HarnessConfig(llm=LLMConfig(model="test"))
    agent = TestAgent(config)
    storage = MemoryStorage()
    runtime = LocalRuntime(workspace_path="/tmp/test-workspace")
    await runtime.setup()

    event_stream = EventStream(session_id="test-session", storage=storage)
    await event_stream.initialize()

    state = State(session_id="test-session")

    controller = AgentController(
        agent=agent,
        event_stream=event_stream,
        state=state,
        storage=storage,
        runtime=runtime,
        max_iterations=10,
    )

    # Add user message
    user_message = MessageAction(content="Hello")
    await event_stream.add_event(user_message, EventSource.USER)

    # Wait a bit for processing
    import asyncio
    await asyncio.sleep(0.1)

    await controller.close()
    await runtime.teardown()


@pytest.mark.asyncio
async def test_controller_handles_finish_action():
    """Test controller handles finish action."""
    from agent_harness.config import HarnessConfig, LLMConfig

    config = HarnessConfig(llm=LLMConfig(model="test"))
    agent = TestAgent(config)
    storage = MemoryStorage()
    runtime = LocalRuntime(workspace_path="/tmp/test-workspace")
    await runtime.setup()

    event_stream = EventStream(session_id="test-session", storage=storage)
    await event_stream.initialize()

    state = State(session_id="test-session")

    controller = AgentController(
        agent=agent,
        event_stream=event_stream,
        state=state,
        storage=storage,
        runtime=runtime,
        max_iterations=10,
    )

    # Add finish action
    finish_action = AgentFinishAction(outputs={"result": "test"})
    await event_stream.add_event(finish_action, EventSource.AGENT)

    # Wait a bit for processing
    import asyncio
    await asyncio.sleep(0.1)

    assert controller.agent_state == AgentState.FINISHED
    assert state.outputs == {"result": "test"}

    await controller.close()
    await runtime.teardown()

