"""Integration tests for AgentHarness."""

import pytest
from agent_harness import AgentHarness, HarnessConfig, LLMConfig
from agent_harness.core.agent import Agent
from agent_harness.core.state import State
from agent_harness.events.action import AgentFinishAction, NullAction
from agent_harness.storage.memory import MemoryStorage
from agent_harness.runtime.local import LocalRuntime


class SimpleTestAgent(Agent):
    """Simple test agent that finishes immediately."""

    async def step(self, state: State):
        """Return finish action."""
        return AgentFinishAction(outputs={"result": "completed"})


@pytest.mark.asyncio
async def test_harness_initialization():
    """Test harness can be initialized."""
    config = HarnessConfig(
        llm=LLMConfig(model="test-model", api_key="test-key")
    )

    harness = AgentHarness(config)
    await harness.initialize()

    assert harness.session_id is not None
    assert harness.controller is not None
    assert harness.event_stream is not None

    await harness.close()


@pytest.mark.asyncio
async def test_harness_run_simple_task():
    """Test harness can run a simple task."""
    config = HarnessConfig(
        llm=LLMConfig(model="test-model", api_key="test-key"),
        max_iterations=10,
    )

    agent = SimpleTestAgent(config)
    storage = MemoryStorage()
    runtime = LocalRuntime(workspace_path="/tmp/test-workspace")

    harness = AgentHarness(
        config=config,
        agent=agent,
        storage=storage,
        runtime=runtime,
    )

    await harness.initialize()

    result = await harness.run("Test task")

    assert result is not None
    assert "result" in result

    await harness.close()


@pytest.mark.asyncio
async def test_harness_context_manager():
    """Test harness works as async context manager."""
    config = HarnessConfig(
        llm=LLMConfig(model="test-model", api_key="test-key"),
        max_iterations=10,
    )

    agent = SimpleTestAgent(config)

    async with AgentHarness(config, agent=agent) as harness:
        result = await harness.run("Test task")
        assert result is not None

    # Harness should be closed after context exit
    assert harness.controller is None or harness.controller._closed

