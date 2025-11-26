---
date: 2025-11-25T21:17:18-05:00
researcher: Auto
git_commit: 2bbe15a329e35f5156cdafcbe63c3fd54978ff98
branch: main
repository: silmari-OpenHands
topic: "Standalone Agent Harness Library: Detaching from OpenHands Repository"
tags: [research, codebase, agents, architecture, library, standalone, decoupling, git-submodule]
status: complete
last_updated: 2025-11-25
last_updated_by: Auto
---

# Research: Standalone Agent Harness Library

**Date**: 2025-11-25T21:17:18-05:00
**Researcher**: Auto
**Git Commit**: 2bbe15a329e35f5156cdafcbe63c3fd54978ff98
**Branch**: main
**Repository**: silmari-OpenHands

## Research Question

How can we detach the agent harness from the OpenHands repository to create a standalone, fully encapsulated library that can be used in other projects without any OpenHands dependencies?

## Summary

This research documents the strategy for extracting the agent harness from OpenHands into a standalone Python library. The goal is to create a fully self-contained library that can be included in other projects (via git submodule, direct copy, or monorepo) without requiring the OpenHands codebase.

**Key Findings:**

1. **Coupling Points**: The harness components are tightly coupled to OpenHands-specific modules including logger, storage, config, and utilities. These need to be abstracted or replaced.

2. **Core Dependencies**: Essential components (EventStream, AgentController, State, BAML integration) have minimal external dependencies and can be extracted with interface abstractions.

3. **OpenHands-Specific Dependencies**: Components like FileStore, logger, config models, and runtime plugins need to be abstracted into interfaces or replaced with generic implementations.

4. **Distribution Strategy**: The library should be structured as a standalone module that can be included in projects via git submodule, direct copy, or monorepo structure.

5. **Migration Path**: A phased approach starting with interface definitions, then component extraction, followed by dependency removal and library structure setup.

## Detailed Findings

### 1. Current Coupling Points

#### 1.1 Import Dependencies

**AgentController** (`openhands/controller/agent_controller.py:27-85`):
```python
from openhands.controller.agent import Agent
from openhands.controller.replay import ReplayManager
from openhands.controller.state.state import State
from openhands.controller.state.state_tracker import StateTracker
from openhands.controller.stuck import StuckDetector
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.exceptions import (...)
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema import AgentState
from openhands.events import (...)
from openhands.storage.files import FileStore
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.server.services.conversation_stats import ConversationStats
```

**EventStream** (`openhands/events/stream.py:10-20`):
```python
from openhands.core.logger import openhands_logger as logger
from openhands.events.event import Event, EventSource
from openhands.events.event_store import EventStore
from openhands.io import json
from openhands.storage import FileStore
from openhands.storage.locations import get_conversation_dir
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.shutdown_listener import should_continue
```

**LLM Module** (`openhands/llm/llm.py:10-33`):
```python
from openhands.core.config import LLMConfig
from openhands.core.exceptions import LLMNoResponseError
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.llm.metrics import Metrics
```

#### 1.2 OpenHands-Specific Components

**Logger** (`openhands/core/logger.py`):
- Custom logging setup with JSON formatting
- OpenHands-specific log levels and handlers
- File-based logging with rotation
- **Action**: Replace with standard Python logging or abstract interface

**Storage** (`openhands/storage/`):
- FileStore interface with multiple implementations (Local, S3, Google Cloud, Memory)
- OpenHands-specific storage locations and paths
- **Action**: Abstract into StorageInterface, provide default implementations

**Config** (`openhands/core/config/`):
- Pydantic models with OpenHands-specific defaults
- TOML parsing with OpenHands conventions
- **Action**: Create simplified config models for harness

**IO Utilities** (`openhands/io/`):
- Custom JSON serialization
- Task reading utilities
- **Action**: Use standard library or abstract

**Runtime** (`openhands/runtime/`):
- OpenHands-specific runtime implementations
- Plugin system for sandbox requirements
- **Action**: Abstract into RuntimeInterface

#### 1.3 External Dependencies

From `pyproject.toml:27-113`, key dependencies for harness:

**Required:**
- `baml-py = "^0.213.0"` - BAML for LLM calls
- `pydantic` - Configuration models (implicit via other deps)
- `python-dotenv` - Environment variable management

**Optional/Replaceable:**
- `litellm` - Not needed (BAML-only)
- `docker` - Only if using Docker runtime
- `fastapi`, `uvicorn` - Only if building web interface
- Most other dependencies are OpenHands-specific

### 2. Standalone Library Structure

#### 2.1 Proposed Package Structure

```
agent-harness/
├── README.md                    # Library documentation
├── LICENSE                      # License file
├── VERSION                      # Version file (e.g., "0.1.0")
├── requirements.txt             # Dependency reference
├── agent_harness/               # Main package
│   ├── __init__.py
│   ├── core/                    # Core harness components
│   │   ├── __init__.py
│   │   ├── harness.py           # Main AgentHarness class
│   │   ├── agent.py             # Agent base class
│   │   ├── controller.py        # AgentController
│   │   └── state.py             # State management
│   ├── events/                  # Event system
│   │   ├── __init__.py
│   │   ├── event.py             # Event base class
│   │   ├── stream.py            # EventStream
│   │   ├── action.py            # Action types
│   │   └── observation.py      # Observation types
│   ├── llm/                     # LLM integration
│   │   ├── __init__.py
│   │   ├── baml_client.py       # BAML client wrapper
│   │   ├── registry.py          # LLM registry
│   │   └── metrics.py           # Metrics tracking
│   ├── interfaces/              # Abstract interfaces
│   │   ├── __init__.py
│   │   ├── runtime.py           # RuntimeInterface
│   │   ├── storage.py           # StorageInterface
│   │   └── logger.py            # LoggerInterface
│   ├── storage/                 # Storage implementations
│   │   ├── __init__.py
│   │   ├── local.py             # Local file storage
│   │   ├── memory.py            # In-memory storage
│   │   └── s3.py                # S3 storage (optional)
│   ├── runtime/                 # Runtime implementations
│   │   ├── __init__.py
│   │   ├── local.py             # Local process runtime
│   │   └── docker.py            # Docker runtime (optional)
│   ├── config/                  # Configuration
│   │   ├── __init__.py
│   │   ├── harness_config.py    # HarnessConfig
│   │   └── llm_config.py        # LLMConfig
│   ├── tools/                   # Tool system
│   │   ├── __init__.py
│   │   ├── registry.py          # Tool registry
│   │   └── base.py              # Tool base class
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── logging.py           # Standard logging setup
│   │   └── async_utils.py       # Async utilities
│   └── baml_schemas/            # BAML schema files
│       ├── completion.baml
│       ├── types.baml
│       └── clients.baml
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_harness.py
│   ├── test_events.py
│   └── test_llm.py
├── examples/                    # Usage examples
│   ├── simple_agent.py
│   └── custom_runtime.py
└── docs/                        # Documentation
    ├── getting-started.md
    ├── api-reference.md
    └── architecture.md
```

#### 2.2 Dependency Configuration

**requirements.txt** (for reference):
```txt
# Core dependencies
baml-py>=0.213.0
pydantic>=2.0.0
python-dotenv>=1.0.0

# Optional dependencies
# docker>=6.0.0  # For Docker runtime
# boto3>=1.26.0  # For S3 storage
```

**pyproject.toml** (optional, for development):
```toml
[project]
name = "agent-harness"
version = "0.1.0"
description = "Standalone agent harness with BAML API layer"
requires-python = ">=3.12"

dependencies = [
    "baml-py>=0.213.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
docker = ["docker>=6.0.0"]
s3 = ["boto3>=1.26.0"]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]
```

**Note**: This is a standalone library, not a pip-installable package. It can be included in projects via:
- Git submodule
- Direct copy/clone
- Monorepo structure

### 3. Interface Abstractions

#### 3.1 Storage Interface

Replace `openhands.storage.FileStore` with abstract interface:

```python
# agent_harness/interfaces/storage.py
from abc import ABC, abstractmethod
from typing import Any

class StorageInterface(ABC):
    """Abstract interface for event and file storage."""

    @abstractmethod
    async def save_event(self, session_id: str, event: dict[str, Any]) -> None:
        """Save event to storage."""
        pass

    @abstractmethod
    async def load_events(self, session_id: str) -> list[dict[str, Any]]:
        """Load events for session."""
        pass

    @abstractmethod
    async def save_file(self, path: str, content: bytes) -> None:
        """Save file to storage."""
        pass

    @abstractmethod
    async def load_file(self, path: str) -> bytes:
        """Load file from storage."""
        pass

    @abstractmethod
    async def list_files(self, prefix: str) -> list[str]:
        """List files with given prefix."""
        pass
```

**Default Implementations:**
- `LocalStorage`: Local file system storage
- `MemoryStorage`: In-memory storage for testing
- `S3Storage`: S3 storage (optional dependency)

#### 3.2 Logger Interface

Replace `openhands.core.logger` with standard logging:

```python
# agent_harness/utils/logging.py
import logging
import sys
from typing import Optional

def setup_logger(
    name: str = "agent_harness",
    level: str = "INFO",
    format_string: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """Setup standard Python logger for harness."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        if json_format:
            from pythonjsonlogger import jsonlogger
            formatter = jsonlogger.JsonFormatter(format_string)
        else:
            format_string = format_string or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            formatter = logging.Formatter(format_string)

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
```

#### 3.3 Runtime Interface

Abstract runtime execution:

```python
# agent_harness/interfaces/runtime.py
from abc import ABC, abstractmethod
from typing import Any

class RuntimeInterface(ABC):
    """Abstract interface for runtime execution."""

    @abstractmethod
    async def execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute an action and return observation."""
        pass

    @abstractmethod
    def get_working_directory(self) -> str:
        """Get current working directory."""
        pass

    @abstractmethod
    async def setup(self) -> None:
        """Initialize runtime environment."""
        pass

    @abstractmethod
    async def teardown(self) -> None:
        """Clean up runtime environment."""
        pass

    @abstractmethod
    async def read_file(self, path: str, start: int = 0, end: int = -1) -> str:
        """Read file content."""
        pass

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write file content."""
        pass
```

**Default Implementations:**
- `LocalRuntime`: Execute commands in local process
- `DockerRuntime`: Execute in Docker container (optional)

#### 3.4 Configuration Models

Create simplified config models:

```python
# agent_harness/config/harness_config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LLMConfig:
    """LLM configuration."""
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    use_baml: bool = True  # Always True for harness

@dataclass
class HarnessConfig:
    """Harness configuration."""
    # LLM Configuration
    llm: LLMConfig

    # Agent Configuration
    agent_name: str = "CodeActAgent"
    tools: list[str] = field(default_factory=lambda: ["cmd", "editor", "think"])

    # Runtime Configuration
    runtime_type: str = "local"  # "local" or "docker"
    workspace_path: str = "./workspace"

    # Storage Configuration
    storage_type: str = "local"  # "local", "memory", "s3"
    storage_path: str = "~/.agent-harness"

    # Execution Configuration
    max_iterations: int = 100
    max_budget: Optional[float] = None
    headless: bool = True
```

### 4. Component Extraction Strategy

#### 4.1 Phase 1: Interface Definition

**Goal**: Define all abstract interfaces without OpenHands dependencies.

**Tasks:**
1. Create `interfaces/` module with abstract base classes
2. Define StorageInterface, RuntimeInterface, LoggerInterface
3. Create simplified config models
4. Document interface contracts

**Dependencies**: None (pure Python ABCs)

#### 4.2 Phase 2: Core Component Extraction

**Goal**: Extract core components with minimal dependencies.

**Components to Extract:**
1. **Event System** (`events/`):
   - `Event` base class
   - `EventStream` implementation
   - Action and Observation types
   - Remove: OpenHands logger, storage locations

2. **State Management** (`core/state.py`):
   - `State` class
   - `StateTracker` class
   - Remove: OpenHands-specific state fields

3. **Agent Base** (`core/agent.py`):
   - `Agent` abstract base class
   - Agent registry
   - Remove: OpenHands config, prompt manager dependencies

4. **LLM Integration** (`llm/`):
   - BAML client wrapper
   - LLM registry
   - Metrics tracking
   - Remove: LiteLLM fallback, OpenHands config

**Dependencies**:
- `baml-py` for BAML integration
- `pydantic` for config validation
- Standard library only

#### 4.3 Phase 3: Implementation Extraction

**Goal**: Extract implementations with interface abstractions.

**Implementations:**
1. **Storage Implementations**:
   - Extract LocalFileStore → LocalStorage
   - Extract InMemoryFileStore → MemoryStorage
   - Create S3Storage (optional)

2. **Runtime Implementations**:
   - Extract LocalRuntime or create new
   - Extract DockerRuntime (optional)

3. **Default Tools**:
   - Extract tool definitions
   - Create tool registry system

**Dependencies**:
- Standard library
- Optional: `docker`, `boto3`

#### 4.4 Phase 4: Dependency Removal

**Goal**: Remove all OpenHands-specific dependencies.

**Replacements:**
1. **Logger**: Replace `openhands.core.logger` with standard `logging`
2. **Storage**: Use StorageInterface implementations
3. **Config**: Use simplified HarnessConfig
4. **IO**: Use standard library `json` instead of `openhands.io.json`
5. **Utils**: Extract or replace async utilities

**Dependencies**: None (all standard library or defined interfaces)

#### 4.5 Phase 5: Library Structure Setup

**Goal**: Set up standalone library structure for inclusion in projects.

**Tasks:**
1. Create library directory structure
2. Set up `__init__.py` with public API
3. Write README and documentation
4. Create dependency requirements file (requirements.txt or pyproject.toml for reference)
5. Set up git repository structure (if using git submodule approach)

### 5. Dependency Analysis

#### 5.1 Required Dependencies

**Core Dependencies:**
- `baml-py>=0.213.0` - BAML for LLM calls (required)
- `pydantic>=2.0.0` - Configuration models (required)
- `python-dotenv>=1.0.0` - Environment variable management (required)

**Python Version:**
- `>=3.12` - Match OpenHands requirement

#### 5.2 Optional Dependencies

**Runtime Options:**
- `docker>=6.0.0` - For Docker runtime (optional)
- `boto3>=1.26.0` - For S3 storage (optional)

**Development:**
- `pytest>=8.0.0` - Testing
- `pytest-asyncio>=0.21.0` - Async testing
- `ruff>=0.1.0` - Linting
- `mypy>=1.0.0` - Type checking

#### 5.3 Removed Dependencies

**Not Needed (BAML-only approach):**
- `litellm` - Replaced by BAML
- `openai` - Not needed directly (BAML handles it)
- `fastapi`, `uvicorn` - Only if building web interface
- `docker` - Optional, only for Docker runtime
- Most OpenHands-specific dependencies

### 6. Import Strategy

#### 6.1 Namespace Changes

**From OpenHands:**
```python
from openhands.controller.agent_controller import AgentController
from openhands.events.stream import EventStream
from openhands.llm.llm_registry import LLMRegistry
```

**To Harness:**
```python
from agent_harness.core.controller import AgentController
from agent_harness.events.stream import EventStream
from agent_harness.llm.registry import LLMRegistry
```

#### 6.2 Public API

**Main Entry Point** (`agent_harness/__init__.py`):
```python
"""Agent Harness - Standalone agent execution framework."""

from agent_harness.core.harness import AgentHarness
from agent_harness.config import HarnessConfig, LLMConfig
from agent_harness.interfaces import RuntimeInterface, StorageInterface

__version__ = "0.1.0"
__all__ = [
    "AgentHarness",
    "HarnessConfig",
    "LLMConfig",
    "RuntimeInterface",
    "StorageInterface",
]
```

#### 6.3 Backward Compatibility

**Not Required**: Since this is a new standalone library, backward compatibility with OpenHands imports is not needed. Users will import from `agent_harness` instead of `openhands`.

### 7. Migration Checklist

#### 7.1 Code Extraction

- [ ] Extract Event system (`events/`)
- [ ] Extract State management (`core/state.py`)
- [ ] Extract Agent base class (`core/agent.py`)
- [ ] Extract AgentController (`core/controller.py`)
- [ ] Extract LLM integration (`llm/`)
- [ ] Extract BAML schemas (`baml_schemas/`)

#### 7.2 Interface Creation

- [ ] Create StorageInterface
- [ ] Create RuntimeInterface
- [ ] Create LoggerInterface (or use standard logging)
- [ ] Create simplified config models

#### 7.3 Implementation Extraction

- [ ] Extract LocalStorage
- [ ] Extract MemoryStorage
- [ ] Create LocalRuntime
- [ ] Extract tool definitions
- [ ] Create tool registry

#### 7.4 Dependency Removal

- [ ] Replace OpenHands logger with standard logging
- [ ] Replace OpenHands storage with StorageInterface
- [ ] Replace OpenHands config with HarnessConfig
- [ ] Replace OpenHands IO with standard library
- [ ] Remove LiteLLM dependencies
- [ ] Remove OpenHands-specific utilities

#### 7.5 Library Structure

- [ ] Create library directory structure
- [ ] Create public API (`__init__.py`)
- [ ] Write README.md with inclusion instructions
- [ ] Write documentation
- [ ] Create requirements.txt (for reference)
- [ ] Set up tests
- [ ] Document inclusion methods (git submodule, direct copy, etc.)

#### 7.6 Testing

- [ ] Extract and adapt existing OpenHands tests for core components
- [ ] Extract and adapt integration tests with BAML
- [ ] Extract and adapt tests for storage implementations
- [ ] Extract and adapt tests for runtime implementations
- [ ] Verify test coverage matches current OpenHands level

### 8. Example Usage

#### 8.1 Basic Usage

```python
from agent_harness import AgentHarness, HarnessConfig, LLMConfig

# Configure harness
llm_config = LLMConfig(
    model="gpt-4o",
    api_key="sk-...",
    use_baml=True
)

config = HarnessConfig(
    llm=llm_config,
    agent_name="CodeActAgent",
    tools=["cmd", "editor", "think"],
    runtime_type="local",
    workspace_path="./workspace",
    storage_type="local",
    storage_path="~/.agent-harness"
)

# Create and run harness
harness = AgentHarness(config)
result = await harness.run("Write a Python script to calculate fibonacci numbers")
print(result)

# Clean up
await harness.close()
```

#### 8.2 Custom Runtime

```python
from agent_harness import AgentHarness, HarnessConfig, LLMConfig
from agent_harness.interfaces import RuntimeInterface

class CustomRuntime(RuntimeInterface):
    """Custom runtime implementation."""

    async def execute_action(self, action: dict) -> dict:
        # Custom execution logic
        pass

    # Implement other required methods...

# Use custom runtime
config = HarnessConfig(
    llm=LLMConfig(model="gpt-4o", api_key="sk-..."),
    runtime=CustomRuntime()
)

harness = AgentHarness(config)
```

#### 8.3 Custom Storage

```python
from agent_harness import AgentHarness, HarnessConfig, LLMConfig
from agent_harness.interfaces import StorageInterface

class CustomStorage(StorageInterface):
    """Custom storage implementation."""

    async def save_event(self, session_id: str, event: dict) -> None:
        # Custom storage logic
        pass

    # Implement other required methods...

# Use custom storage
config = HarnessConfig(
    llm=LLMConfig(model="gpt-4o", api_key="sk-..."),
    storage=CustomStorage()
)

harness = AgentHarness(config)
```

### 9. Library Inclusion Methods

Since the library is not distributed via pip, it can be included in projects using several methods:

#### 9.1 Git Submodule

**Add as submodule:**
```bash
git submodule add https://github.com/your-org/agent-harness.git libs/agent-harness
git submodule update --init --recursive
```

**Use in project:**
```python
import sys
sys.path.insert(0, 'libs/agent-harness')

from agent_harness import AgentHarness, HarnessConfig
```

**Update submodule:**
```bash
git submodule update --remote libs/agent-harness
```

#### 9.2 Direct Copy (Primary Method)

**Copy directory into project:**
```bash
# Copy the agent-harness directory into your project
cp -r /path/to/agent-harness libs/agent-harness

# Or clone and copy
git clone https://github.com/your-org/agent-harness.git /tmp/agent-harness
cp -r /tmp/agent-harness libs/agent-harness
```

**Use in project:**
```python
import sys
sys.path.insert(0, 'libs/agent-harness')

from agent_harness import AgentHarness, HarnessConfig
```

**Note**: This is the primary distribution method. The library is designed to be directly copied into projects, allowing for easy customization and version control within each project.

#### 9.3 Monorepo Structure

**In monorepo:**
```
monorepo/
├── agent-harness/          # Standalone library
│   ├── agent_harness/
│   └── ...
├── project-a/              # Project using harness
│   └── main.py
└── project-b/              # Another project using harness
    └── main.py
```

**Use in project:**
```python
import sys
sys.path.insert(0, '../agent-harness')

from agent_harness import AgentHarness, HarnessConfig
```

#### 9.4 Python Path Configuration

**Add to PYTHONPATH:**
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/agent-harness"
```

**Or in project's setup:**
```python
# setup.py or project initialization
import sys
from pathlib import Path

harness_path = Path(__file__).parent / 'libs' / 'agent-harness'
sys.path.insert(0, str(harness_path))
```

### 10. Benefits of Standalone Library

#### 10.1 Independence

- **No OpenHands Dependency**: Can be used in any Python project
- **Minimal Dependencies**: Only BAML and standard library for core functionality
- **Clear Boundaries**: Well-defined interfaces separate concerns

#### 10.2 Reusability

- **Flexible Inclusion**: Can be included via git submodule, direct copy, or monorepo
- **Version Control**: Independent semantic versioning (starting at 0.1.0) tracked via git tags and VERSION file
- **Cross-Project**: Use in multiple projects by including the library directory

#### 10.3 Maintainability

- **Focused Scope**: Only agent harness functionality
- **Simpler Codebase**: No OpenHands-specific complexity
- **Clear API**: Well-defined public interface

#### 10.4 Extensibility

- **Plugin System**: Easy to add custom runtimes and storage
- **Interface-Based**: Extend via interfaces, not inheritance
- **Modular Design**: Components can be replaced independently

## Code References

### Current Coupling Points
- `openhands/controller/agent_controller.py:27-85` - Import dependencies
- `openhands/events/stream.py:10-20` - EventStream dependencies
- `openhands/llm/llm.py:10-33` - LLM module dependencies
- `openhands/core/logger.py:1-50` - Logger implementation
- `openhands/storage/__init__.py:1-57` - Storage factory

### Dependencies
- `pyproject.toml:27-113` - Current dependency list
- `openhands/core/config/llm_config.py:12-199` - LLMConfig with OpenHands defaults
- `openhands/io/__init__.py:1-10` - IO utilities

## Architecture Documentation

### Current Architecture

OpenHands uses a monolithic structure where:
- All components are in `openhands/` namespace
- Tight coupling via shared utilities (logger, storage, config)
- OpenHands-specific defaults and conventions throughout

### Proposed Standalone Architecture

The standalone library will:
1. **Use Standard Interfaces**: Abstract interfaces for all external dependencies
2. **Minimal Dependencies**: Only BAML and standard library for core
3. **Clear Public API**: Well-defined entry points and exports
4. **Modular Design**: Components can be replaced via interfaces
5. **No OpenHands References**: Completely independent codebase

### Component Dependencies

```
AgentHarness
├── Core Components (no external deps)
│   ├── EventStream
│   ├── AgentController
│   ├── State
│   └── Agent
├── LLM Integration (BAML only)
│   ├── BAMLClient
│   └── LLMRegistry
├── Interfaces (ABCs only)
│   ├── RuntimeInterface
│   ├── StorageInterface
│   └── LoggerInterface
└── Implementations (optional)
    ├── LocalStorage
    ├── MemoryStorage
    ├── LocalRuntime
    └── DockerRuntime (optional)
```

## Historical Context (from thoughts/)

- `thoughts/shared/research/2025-11-25-encapsulated-agent-harness-baml.md` - Previous research on harness design with BAML API layer
- `thoughts/shared/research/2025-11-25-agent-architecture-research.md` - Foundation research on agent architecture

## Related Research

- `thoughts/shared/research/2025-11-25-encapsulated-agent-harness-baml.md` - Harness design with BAML
- `thoughts/shared/research/2025-11-25-agent-architecture-research.md` - Agent architecture foundation

## Open Questions - Resolved

1. **Versioning Strategy**: ✅ **New versioning** - The library will use independent versioning (e.g., semantic versioning starting at 0.1.0) separate from OpenHands. Version will be tracked via git tags and a `VERSION` file in the library root.

2. **BAML Schema Updates**: ✅ **Separate from OpenHands** - BAML schemas will be maintained independently in the standalone library. Updates will be versioned along with the library, not tied to OpenHands releases.

3. **Tool Compatibility**: ✅ **Recommendation**: Tools should be designed with runtime-agnostic interfaces. Each tool should define its requirements (e.g., file system access, command execution) and the runtime interface should provide these capabilities. Tools can check runtime capabilities at initialization and adapt behavior accordingly. This allows tools to work across different runtime implementations while maintaining flexibility.

4. **Testing Strategy**: ✅ **Leverage existing testing coverage** - Extract and adapt existing OpenHands tests for the harness components. Focus on:
   - Unit tests for core components (EventStream, State, AgentController)
   - Integration tests with BAML
   - Runtime implementation tests
   - Tool compatibility tests

5. **Documentation**: ✅ **Do later** - Initial release will have minimal documentation (README with basic usage). Comprehensive documentation can be added in subsequent iterations.

6. **Distribution Method**: ✅ **Direct copy** - The library will be distributed as a directory that can be directly copied into projects. Users will include it via:
   ```python
   import sys
   sys.path.insert(0, 'path/to/agent-harness')
   from agent_harness import AgentHarness
   ```

7. **Backward Compatibility**: ✅ **No** - No backward compatibility with OpenHands will be maintained. The library is a fresh start with a clean API. Users migrating from OpenHands will need to adapt their code to the new interface.

8. **Plugin System**: ✅ **Match current level** - The plugin system will match the current extensibility level in OpenHands:
   - Runtime implementations via RuntimeInterface
   - Storage implementations via StorageInterface
   - Tool registration via ToolRegistry
   - Custom agents via Agent base class

## Recommendations

### Immediate Next Steps

1. **Create Interface Definitions**: Start with abstract interfaces for Storage, Runtime, Logger
2. **Extract Core Components**: Begin with Event system and State management
3. **Set Up Library Structure**: Create initial library directory structure
4. **Create Minimal Implementation**: Build LocalStorage and LocalRuntime as defaults
5. **Extract and Adapt Tests**: Extract relevant tests from OpenHands and adapt for standalone library

### Long-Term Considerations

1. **Plugin Architecture**: Maintain current extensibility level (RuntimeInterface, StorageInterface, ToolRegistry, Agent base class)
2. **Documentation**: Add comprehensive documentation and examples in later iterations
3. **CI/CD Pipeline**: Set up automated testing (if using git repository)
4. **Version Management**: Use independent semantic versioning (starting at 0.1.0) tracked via git tags and VERSION file
5. **Direct Copy Distribution**: Document best practices for including library via direct copy method

