# Agent Harness

A standalone, fully encapsulated agent execution framework extracted from OpenHands. The library is completely independent and can be included in other projects via direct copy, git submodule, or monorepo structure.

## Features

- **Standalone**: Zero dependencies on OpenHands codebase
- **BAML-Only**: Uses BAML as the exclusive LLM API layer
- **Interface-Based**: Storage, Runtime, and Logger use abstract interfaces
- **Testable**: Comprehensive test coverage for all components
- **Flexible**: Can be integrated into any Python project

## Installation

### Option 1: Direct Copy

Copy the `agent-harness` directory into your project:

```bash
cp -r agent-harness /path/to/your/project/libs/
```

Then add to your Python path:

```python
import sys
sys.path.insert(0, '/path/to/your/project/libs/agent-harness')
from agent_harness import AgentHarness, HarnessConfig, LLMConfig
```

### Option 2: Git Submodule

```bash
git submodule add <repository-url> libs/agent-harness
```

### Option 3: Monorepo Structure

Include `agent-harness` as a subdirectory in your monorepo.

## Quick Start

### 1. Install Dependencies

```bash
cd agent-harness
pip install -r requirements.txt
```

### 2. Configure BAML Client

Generate the BAML client from schemas:

```bash
cd agent-harness/agent_harness
baml update-client
```

This generates the `baml_client/` directory from `baml_src/` schemas.

### 3. Set Up Environment Variables

Create a `.env` file (or set environment variables):

```bash
# .env file
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://api.openai.com/v1  # Optional, depends on provider
LLM_MODEL=gpt-4  # Optional, can be set in code
```

### 4. Create an Agent

```python
from agent_harness import AgentHarness, HarnessConfig, LLMConfig
from agent_harness.core.agent import Agent
from agent_harness.core.state import State
from agent_harness.events.action import AgentFinishAction

class SimpleAgent(Agent):
    """Simple agent that completes immediately."""
    
    async def step(self, state: State):
        """Execute one step of the agent."""
        # Your agent logic here
        return AgentFinishAction(outputs={"result": "completed"})

# Configure the harness
config = HarnessConfig(
    llm=LLMConfig(
        model="gpt-4",
        api_key="your-api-key",  # Or load from .env
    ),
    max_iterations=100,
)

# Create and run the harness
async def main():
    agent = SimpleAgent(config)
    
    async with AgentHarness(config, agent=agent) as harness:
        result = await harness.run("Your task description here")
        print(f"Result: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Configuration

### LLMConfig

Configure your LLM provider:

```python
from agent_harness import LLMConfig

# Option 1: Direct configuration
llm_config = LLMConfig(
    model="gpt-4",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",  # Optional
    temperature=0.0,
    max_tokens=1000,
)

# Option 2: Load from environment variables
llm_config = LLMConfig(model="gpt-4")  # api_key loaded from LLM_API_KEY env var
```

### HarnessConfig

Configure the harness:

```python
from agent_harness import HarnessConfig, LLMConfig

config = HarnessConfig(
    llm=LLMConfig(model="gpt-4"),
    agent_name="CodeActAgent",  # Name of registered agent
    tools=["cmd", "editor", "think"],  # Available tools
    workspace_path="./workspace",  # Working directory
    storage_path="~/.agent-harness",  # Storage location
    max_iterations=100,  # Maximum agent steps
    headless=True,  # Run without UI
)
```

## Architecture

### Components

- **AgentHarness**: Main entry point for running agents
- **AgentController**: Orchestrates agent execution
- **EventStream**: Manages event persistence
- **State**: Tracks agent state
- **LLM**: BAML-based LLM integration
- **Storage**: Abstract storage interface (LocalStorage, MemoryStorage)
- **Runtime**: Abstract runtime interface (LocalRuntime)

### Interfaces

The library uses abstract interfaces for flexibility:

- `StorageInterface`: Event and file storage
- `RuntimeInterface`: Action execution environment
- `LoggerInterface`: Standard Python logging

## Examples

See `examples/simple_agent.py` for a complete working example.

## Development

### Running Tests

```bash
cd agent-harness
pytest tests/ -v
```

### Type Checking

```bash
mypy agent_harness/
```

### Linting

```bash
ruff check agent_harness/
```

## Requirements

- Python 3.8+
- `baml-py>=0.214.0`
- `pydantic>=2.0.0`
- `python-dotenv>=1.0.0` (optional, for .env file support)

## BAML Integration

The library uses BAML (Binary Application Markup Language) for LLM integration. BAML schemas are located in `agent_harness/baml_src/`:

- `completion.baml`: Main completion function
- `types.baml`: Type definitions
- `clients.baml`: LLM client configurations
- `generators.baml`: Code generation settings

The BAML client is auto-generated in `agent_harness/baml_client/` when you run `baml update-client`.

## Storage

### LocalStorage

Default file system storage:

```python
from agent_harness.storage.local import LocalStorage

storage = LocalStorage(base_path="~/.agent-harness")
```

### MemoryStorage

In-memory storage for testing:

```python
from agent_harness.storage.memory import MemoryStorage

storage = MemoryStorage()
```

## Runtime

### LocalRuntime

Local process runtime:

```python
from agent_harness.runtime.local import LocalRuntime

runtime = LocalRuntime(workspace_path="./workspace")
```

## License

See LICENSE file for details.

## Contributing

This is a standalone library extracted from OpenHands. For contributions, please follow the same guidelines as the main OpenHands project.

## Version

Current version: 0.1.0

