"""Tests for agent base class."""

import pytest
from agent_harness.core.agent import Agent
from agent_harness.config import HarnessConfig, LLMConfig


class TestAgent(Agent):
    """Test agent implementation."""

    async def step(self, event):
        """Test step implementation."""
        return {"type": "observation", "content": "test"}


@pytest.mark.asyncio
async def test_agent_creation():
    """Test creating agent."""
    config = HarnessConfig(llm=LLMConfig(model="test"))
    agent = TestAgent(config)
    assert agent.name == "TestAgent"
    assert not agent.complete


@pytest.mark.asyncio
async def test_agent_registration():
    """Test agent registration."""
    @Agent.register("CustomTestAgent")
    class CustomAgent(Agent):
        async def step(self, event):
            return {"type": "observation"}

    assert "CustomTestAgent" in Agent._registry
    assert Agent.get_agent_class("CustomTestAgent") == CustomAgent


@pytest.mark.asyncio
async def test_agent_get_agent_class():
    """Test getting agent class."""
    @Agent.register()
    class AnotherAgent(Agent):
        async def step(self, event):
            return {"type": "observation"}

    agent_class = Agent.get_agent_class("AnotherAgent")
    assert agent_class == AnotherAgent

    with pytest.raises(ValueError):
        Agent.get_agent_class("NonexistentAgent")

