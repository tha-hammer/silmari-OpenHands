"""Simple example of using AgentHarness.

This example demonstrates:
1. Creating a simple agent
2. Configuring the harness
3. Running a task
4. Using async context manager
"""

import asyncio
from agent_harness import AgentHarness, HarnessConfig, LLMConfig
from agent_harness.core.agent import Agent
from agent_harness.core.state import State
from agent_harness.events.action import AgentFinishAction


class SimpleAgent(Agent):
    """Simple agent that completes immediately with a result."""

    async def step(self, state: State) -> AgentFinishAction:
        """Execute one step of the agent.
        
        In a real agent, this would:
        1. Analyze the current state
        2. Decide on an action
        3. Return the action to execute
        
        For this example, we just finish immediately.
        """
        return AgentFinishAction(outputs={"result": "Task completed successfully"})


async def main():
    """Main example function."""
    # Configure LLM (API key can be loaded from .env file or environment variables)
    llm_config = LLMConfig(
        model="gpt-4",  # Or any supported model
        # api_key="your-api-key",  # Optional: set here or via LLM_API_KEY env var
        # base_url="https://api.openai.com/v1",  # Optional: set here or via LLM_BASE_URL env var
    )

    # Configure harness
    config = HarnessConfig(
        llm=llm_config,
        max_iterations=10,  # Maximum number of agent steps
        workspace_path="./example-workspace",  # Working directory
        storage_path="~/.agent-harness-example",  # Storage location
    )

    # Create agent
    agent = SimpleAgent(config)

    # Use async context manager for automatic cleanup
    async with AgentHarness(config, agent=agent) as harness:
        # Run a task
        task = "Complete this example task"
        print(f"Running task: {task}")
        
        result = await harness.run(task)
        
        print(f"Task completed!")
        print(f"Result: {result}")
        print(f"Session ID: {harness.session_id}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())

