---
date: 2025-11-25T21:26:39-05:00
git_commit: 2bbe15a329e35f5156cdafcbe63c3fd54978ff98
branch: main
repository: silmari-OpenHands
topic: "Standalone Agent Harness Library Implementation"
tags: [implementation-plan, agents, harness, library, standalone, tdd]
status: in-progress
last_updated: 2025-11-26
implementation_branch: standalone-harness-library
checkpoints:
  - phase1: 2accaa20
  - phase2: 6c6b40f5
  - phase3: e55e33d4
---

```
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│     STANDALONE AGENT HARNESS LIBRARY                         │
│     Implementation Plan                                      │
│                                                               │
│     Status: In Progress (Phases 1-3 Complete)               │
│     Date: 2025-11-25                                         │
│     Last Updated: 2025-11-26                                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

# Standalone Agent Harness Library Implementation Plan

## Overview

This plan implements a standalone, fully encapsulated agent harness library extracted from OpenHands. The library will be completely independent, using BAML as the exclusive LLM API layer, and can be included in other projects via direct copy, git submodule, or monorepo structure.

**Key Principles:**
- **TDD Approach**: Each phase ends with testable, verifiable components
- **Incremental Extraction**: Build interfaces first, then extract components
- **Zero OpenHands Dependencies**: Complete decoupling from OpenHands codebase
- **BAML-Only**: No LiteLLM fallback, BAML is the exclusive API layer

## Current State Analysis

### Existing Components

| Component | Location | Dependencies | Status |
|-----------|----------|--------------|--------|
| AgentController | `openhands/controller/agent_controller.py` | OpenHands logger, storage, config | ✅ Extractable |
| EventStream | `openhands/events/stream.py` | OpenHands logger, storage, IO | ✅ Extractable |
| State | `openhands/controller/state/state.py` | OpenHands schema, storage | ✅ Extractable |
| LLM Integration | `openhands/llm/llm.py` | BAML, OpenHands config | ✅ Extractable |
| BAML Schemas | `openhands/llm/baml_src/` | None | ✅ Ready |

### Coupling Points

**Critical Dependencies to Remove:**
- `openhands.core.logger` → Replace with standard `logging`
- `openhands.storage.files.FileStore` → Abstract to `StorageInterface`
- `openhands.core.config` → Create simplified `HarnessConfig`
- `openhands.io.json` → Use standard library `json`
- `openhands.runtime` → Abstract to `RuntimeInterface`

### Key Discoveries

| Discovery | Impact | Action |
|-----------|--------|--------|
| BAML already integrated | ✅ Can use existing BAML schemas | Extract and adapt |
| Event system is self-contained | ✅ Minimal dependencies | Extract with interface abstractions |
| State management is complex | ⚠️ Needs careful extraction | Simplify for standalone use |
| Storage is abstracted | ✅ Can create interface easily | Define `StorageInterface` first |
| Runtime is abstracted | ✅ Can create interface easily | Define `RuntimeInterface` first |

## Desired End State

After implementation, we will have:

1. **Standalone Library Structure**: `agent-harness/` directory with complete package structure
2. **Zero OpenHands Dependencies**: All imports use `agent_harness.*` namespace
3. **BAML-Only LLM**: All LLM calls go through BAML, no LiteLLM fallback
4. **Interface-Based Design**: Storage, Runtime, Logger use abstract interfaces
5. **Testable Components**: Each component has comprehensive test coverage
6. **Includable in Projects**: Can be copied into any Python project

### Verification

**Automated:**
- All tests pass: `pytest tests/`
- No OpenHands imports: `grep -r "from openhands" agent_harness/` returns nothing
- Type checking passes: `mypy agent_harness/`
- Linting passes: `ruff check agent_harness/`

**Manual:**
- Library can be imported in a fresh Python project
- Example usage works end-to-end
- BAML client generates successfully

## What We're NOT Doing

| Not Doing | Why | Alternative |
|-----------|-----|-------------|
| Pip-installable package | Distribution via direct copy | Include via `sys.path` |
| Backward compatibility with OpenHands | Clean break for simplicity | Migration guide in docs |
| LiteLLM fallback | BAML-only design | BAML handles all providers |
| Web UI/server components | Out of scope | Focus on core harness |
| Docker runtime (initially) | Phase 1 focuses on local | Add in later phase |

## Implementation Approach

**Strategy**: TDD with incremental extraction
1. **Phase 1**: Define interfaces + tests (RED)
2. **Phase 2**: Implement interfaces + tests pass (GREEN)
3. **Phase 3**: Extract core components + tests (REFACTOR)
4. **Phase 4**: Remove dependencies + integration tests
5. **Phase 5**: Library structure + documentation

Each phase ends with **verifiable tests** that can be run independently.

---

## Phase 1: Interface Definitions & Test Structure

### Overview
Define all abstract interfaces and write tests that will guide implementation. This phase establishes the contract for all components.

### Changes Required

#### 1. Create Library Structure
**File**: `agent-harness/agent_harness/__init__.py` (new)
**Changes**: Create package structure

```python
"""Agent Harness - Standalone agent execution framework."""

__version__ = "0.1.0"
```

**File**: `agent-harness/agent_harness/interfaces/__init__.py` (new)
**Changes**: Export interfaces

```python
from agent_harness.interfaces.storage import StorageInterface
from agent_harness.interfaces.runtime import RuntimeInterface
from agent_harness.interfaces.logger import LoggerInterface

__all__ = ["StorageInterface", "RuntimeInterface", "LoggerInterface"]
```

#### 2. Define Storage Interface
**File**: `agent-harness/agent_harness/interfaces/storage.py` (new)
**Changes**: Abstract storage interface

```python
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

#### 3. Define Runtime Interface
**File**: `agent-harness/agent_harness/interfaces/runtime.py` (new)
**Changes**: Abstract runtime interface

```python
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

#### 4. Define Logger Interface (Standard Logging)
**File**: `agent-harness/agent_harness/utils/logging.py` (new)
**Changes**: Standard logging setup

```python
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

#### 5. Define Configuration Models
**File**: `agent-harness/agent_harness/config/__init__.py` (new)
**Changes**: Configuration models

```python
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
    llm: LLMConfig
    agent_name: str = "CodeActAgent"
    tools: list[str] = field(default_factory=lambda: ["cmd", "editor", "think"])
    runtime_type: str = "local"
    workspace_path: str = "./workspace"
    storage_type: str = "local"
    storage_path: str = "~/.agent-harness"
    max_iterations: int = 100
    max_budget: Optional[float] = None
    headless: bool = True
```

#### 6. Write Interface Tests (TDD - RED)
**File**: `agent-harness/tests/test_interfaces.py` (new)
**Changes**: Tests for interfaces (will fail until Phase 2)

```python
"""Tests for interface contracts."""

import pytest
from agent_harness.interfaces import StorageInterface, RuntimeInterface

def test_storage_interface_contract():
    """Test that StorageInterface defines required methods."""
    # This will fail until we have implementations
    assert hasattr(StorageInterface, 'save_event')
    assert hasattr(StorageInterface, 'load_events')
    # ... more assertions

def test_runtime_interface_contract():
    """Test that RuntimeInterface defines required methods."""
    assert hasattr(RuntimeInterface, 'execute_action')
    assert hasattr(RuntimeInterface, 'get_working_directory')
    # ... more assertions

@pytest.mark.asyncio
async def test_storage_interface_cannot_instantiate():
    """Test that abstract interface cannot be instantiated."""
    with pytest.raises(TypeError):
        StorageInterface()  # Should fail - abstract class

@pytest.mark.asyncio
async def test_runtime_interface_cannot_instantiate():
    """Test that abstract interface cannot be instantiated."""
    with pytest.raises(TypeError):
        RuntimeInterface()  # Should fail - abstract class
```

### Success Criteria

#### Automated Verification:
- [x] Interface definitions exist: `ls agent-harness/agent_harness/interfaces/`
- [x] Interface tests exist: `ls agent-harness/tests/test_interfaces.py`
- [x] Interface tests fail (RED): `pytest agent-harness/tests/test_interfaces.py -v` (expected to fail)
- [x] Type checking passes: `mypy agent-harness/agent_harness/interfaces/`
- [x] No OpenHands imports: `grep -r "from openhands" agent-harness/agent_harness/interfaces/` returns nothing

#### Manual Verification:
- [x] Interfaces are well-documented with docstrings
- [x] Interface structure matches research document
- [x] Configuration models are simple and clear

---

## Phase 2: Interface Implementations & Tests (GREEN)

### Overview
Implement concrete classes for all interfaces and make Phase 1 tests pass. This establishes the foundation for component extraction.

### Changes Required

#### 1. Implement LocalStorage
**File**: `agent-harness/agent_harness/storage/local.py` (new)
**Changes**: Local file system storage implementation

```python
"""Local file system storage implementation."""

import json
import os
from pathlib import Path
from typing import Any

from agent_harness.interfaces.storage import StorageInterface

class LocalStorage(StorageInterface):
    """Local file system storage implementation."""

    def __init__(self, base_path: str = "~/.agent-harness"):
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save_event(self, session_id: str, event: dict[str, Any]) -> None:
        """Save event to local file system."""
        events_dir = self.base_path / "sessions" / session_id / "events"
        events_dir.mkdir(parents=True, exist_ok=True)

        event_file = events_dir / f"{event.get('id', 'event')}.json"
        with open(event_file, 'w') as f:
            json.dump(event, f)

    async def load_events(self, session_id: str) -> list[dict[str, Any]]:
        """Load events from local file system."""
        events_dir = self.base_path / "sessions" / session_id / "events"
        if not events_dir.exists():
            return []

        events = []
        for event_file in sorted(events_dir.glob("*.json")):
            with open(event_file, 'r') as f:
                events.append(json.load(f))

        return events

    # ... implement other methods
```

#### 2. Implement MemoryStorage
**File**: `agent-harness/agent_harness/storage/memory.py` (new)
**Changes**: In-memory storage for testing

```python
"""In-memory storage implementation for testing."""

from typing import Any

from agent_harness.interfaces.storage import StorageInterface

class MemoryStorage(StorageInterface):
    """In-memory storage implementation."""

    def __init__(self):
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._files: dict[str, bytes] = {}

    async def save_event(self, session_id: str, event: dict[str, Any]) -> None:
        """Save event to memory."""
        if session_id not in self._events:
            self._events[session_id] = []
        self._events[session_id].append(event)

    async def load_events(self, session_id: str) -> list[dict[str, Any]]:
        """Load events from memory."""
        return self._events.get(session_id, [])

    # ... implement other methods
```

#### 3. Implement LocalRuntime
**File**: `agent-harness/agent_harness/runtime/local.py` (new)
**Changes**: Local process runtime implementation

```python
"""Local process runtime implementation."""

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any

from agent_harness.interfaces.runtime import RuntimeInterface

class LocalRuntime(RuntimeInterface):
    """Local process runtime implementation."""

    def __init__(self, workspace_path: str = "./workspace"):
        self.workspace_path = Path(workspace_path).resolve()
        self.workspace_path.mkdir(parents=True, exist_ok=True)

    async def setup(self) -> None:
        """Initialize runtime environment."""
        # Ensure workspace exists
        self.workspace_path.mkdir(parents=True, exist_ok=True)

    async def teardown(self) -> None:
        """Clean up runtime environment."""
        # Nothing to clean up for local runtime
        pass

    def get_working_directory(self) -> str:
        """Get current working directory."""
        return str(self.workspace_path)

    async def execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute action in local process."""
        action_type = action.get("type")

        if action_type == "cmd":
            command = action.get("command", "")
            result = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_path
            )
            stdout, stderr = await result.communicate()

            return {
                "type": "observation",
                "content": stdout.decode(),
                "error": stderr.decode() if result.returncode != 0 else None,
                "exit_code": result.returncode
            }

        # Handle other action types...
        return {"type": "observation", "content": ""}

    async def read_file(self, path: str, start: int = 0, end: int = -1) -> str:
        """Read file content."""
        file_path = self.workspace_path / path
        with open(file_path, 'r') as f:
            content = f.read()
            if end == -1:
                return content[start:]
            return content[start:end]

    async def write_file(self, path: str, content: str) -> None:
        """Write file content."""
        file_path = self.workspace_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
```

#### 4. Write Implementation Tests (TDD - GREEN)
**File**: `agent-harness/tests/test_storage_implementations.py` (new)
**Changes**: Tests for storage implementations

```python
"""Tests for storage implementations."""

import pytest
from agent_harness.storage.local import LocalStorage
from agent_harness.storage.memory import MemoryStorage

@pytest.mark.asyncio
async def test_local_storage_save_and_load_event():
    """Test LocalStorage can save and load events."""
    storage = LocalStorage(base_path="/tmp/test-harness")
    session_id = "test-session"

    event = {"id": "event-1", "type": "action", "content": "test"}
    await storage.save_event(session_id, event)

    events = await storage.load_events(session_id)
    assert len(events) == 1
    assert events[0]["id"] == "event-1"

@pytest.mark.asyncio
async def test_memory_storage_save_and_load_event():
    """Test MemoryStorage can save and load events."""
    storage = MemoryStorage()
    session_id = "test-session"

    event = {"id": "event-1", "type": "action", "content": "test"}
    await storage.save_event(session_id, event)

    events = await storage.load_events(session_id)
    assert len(events) == 1
    assert events[0]["id"] == "event-1"
```

**File**: `agent-harness/tests/test_runtime_implementations.py` (new)
**Changes**: Tests for runtime implementations

```python
"""Tests for runtime implementations."""

import pytest
from agent_harness.runtime.local import LocalRuntime

@pytest.mark.asyncio
async def test_local_runtime_setup_and_teardown():
    """Test LocalRuntime setup and teardown."""
    runtime = LocalRuntime(workspace_path="/tmp/test-workspace")
    await runtime.setup()

    assert runtime.get_working_directory() == "/tmp/test-workspace"

    await runtime.teardown()

@pytest.mark.asyncio
async def test_local_runtime_execute_cmd_action():
    """Test LocalRuntime can execute command actions."""
    runtime = LocalRuntime(workspace_path="/tmp/test-workspace")
    await runtime.setup()

    action = {"type": "cmd", "command": "echo 'hello'"}
    observation = await runtime.execute_action(action)

    assert observation["type"] == "observation"
    assert "hello" in observation["content"]

    await runtime.teardown()
```

### Success Criteria

#### Automated Verification:
- [x] All Phase 1 interface tests pass: `pytest agent-harness/tests/test_interfaces.py -v`
- [x] Storage implementation tests pass: `pytest agent-harness/tests/test_storage_implementations.py -v`
- [x] Runtime implementation tests pass: `pytest agent-harness/tests/test_runtime_implementations.py -v`
- [x] Type checking passes: `mypy agent-harness/agent_harness/storage/ agent-harness/agent_harness/runtime/`
- [x] Linting passes: `ruff check agent-harness/agent_harness/storage/ agent-harness/agent_harness/runtime/`
- [x] No OpenHands imports: `grep -r "from openhands" agent-harness/agent_harness/storage/ agent-harness/agent_harness/runtime/` returns nothing

#### Manual Verification:
- [x] LocalStorage creates directories correctly
- [x] MemoryStorage works for testing scenarios
- [x] LocalRuntime can execute basic commands

---

## Phase 3: Core Component Extraction (REFACTOR)

### Overview
Extract core components (Event, EventStream, State, Agent base) from OpenHands and adapt them to use interfaces. Write tests for each component.

### Changes Required

#### 1. Extract Event System
**File**: `agent-harness/agent_harness/events/__init__.py` (new)
**File**: `agent-harness/agent_harness/events/event.py` (new)
**File**: `agent-harness/agent_harness/events/stream.py` (new)
**Changes**: Extract and adapt event system

**Key Adaptations:**
- Replace `openhands.core.logger` with standard `logging`
- Replace `openhands.storage.FileStore` with `StorageInterface`
- Replace `openhands.io.json` with standard `json`
- Remove OpenHands-specific event types (keep core ones)

#### 2. Extract State Management
**File**: `agent-harness/agent_harness/core/state.py` (new)
**Changes**: Extract and simplify State class

**Key Adaptations:**
- Remove OpenHands-specific state fields
- Use `StorageInterface` for persistence
- Simplify to essential state tracking

#### 3. Extract Agent Base Class
**File**: `agent-harness/agent_harness/core/agent.py` (new)
**Changes**: Extract Agent abstract base class

**Key Adaptations:**
- Remove OpenHands config dependencies
- Use `HarnessConfig` instead
- Abstract tool registration

#### 4. Extract BAML Schemas
**File**: `agent-harness/agent_harness/baml_schemas/completion.baml` (new)
**File**: `agent-harness/agent_harness/baml_schemas/types.baml` (new)
**Changes**: Copy and adapt BAML schemas from OpenHands

#### 5. Write Component Tests
**File**: `agent-harness/tests/test_events.py` (new)
**File**: `agent-harness/tests/test_state.py` (new)
**File**: `agent-harness/tests/test_agent.py` (new)
**Changes**: Comprehensive tests for each component

### Success Criteria

#### Automated Verification:
- [x] Event system tests pass: `pytest agent-harness/tests/test_events.py -v`
- [x] State tests pass: `pytest agent-harness/tests/test_state.py -v`
- [x] Agent base tests pass: `pytest agent-harness/tests/test_agent.py -v`
- [ ] BAML client generates: `cd agent-harness/agent_harness/baml_schemas && baml update-client` (deferred to Phase 4)
- [x] Type checking passes: `mypy agent-harness/agent_harness/events/ agent-harness/agent_harness/core/`
- [x] No OpenHands imports: `grep -r "from openhands" agent-harness/agent_harness/events/ agent-harness/agent_harness/core/` returns nothing

#### Manual Verification:
- [x] Events can be saved and loaded
- [x] State persists correctly
- [x] Agent base class can be subclassed

---

## Phase 4: AgentController & LLM Integration

### Overview
Extract AgentController and LLM integration, connecting all components together. This is the integration phase.

### Changes Required

#### 1. Extract AgentController
**File**: `agent-harness/agent_harness/core/controller.py` (new)
**Changes**: Extract and adapt AgentController

**Key Adaptations:**
- Use `StorageInterface` instead of `FileStore`
- Use standard `logging` instead of OpenHands logger
- Use `RuntimeInterface` instead of OpenHands runtime
- Use `HarnessConfig` instead of OpenHands config

#### 2. Extract LLM Integration
**File**: `agent-harness/agent_harness/llm/__init__.py` (new)
**File**: `agent-harness/agent_harness/llm/baml_client.py` (new)
**File**: `agent-harness/agent_harness/llm/registry.py` (new)
**Changes**: Extract BAML-only LLM integration

**Key Adaptations:**
- Remove LiteLLM fallback (BAML-only)
- Use `HarnessConfig` for LLM configuration
- Use standard `logging`

#### 3. Create Main Harness Class
**File**: `agent-harness/agent_harness/core/harness.py` (new)
**Changes**: Main entry point for the library

```python
"""Main AgentHarness class."""

from agent_harness.config import HarnessConfig
from agent_harness.core.controller import AgentController
from agent_harness.events.stream import EventStream
from agent_harness.storage.local import LocalStorage
from agent_harness.runtime.local import LocalRuntime

class AgentHarness:
    """Main harness class for running agents."""

    def __init__(self, config: HarnessConfig):
        self.config = config
        self.storage = self._create_storage()
        self.runtime = self._create_runtime()
        self.event_stream = EventStream(session_id="default", storage=self.storage)
        # ... initialize controller

    async def run(self, task: str) -> dict:
        """Run agent with given task."""
        # ... implementation

    async def close(self) -> None:
        """Clean up resources."""
        # ... implementation
```

#### 4. Write Integration Tests
**File**: `agent-harness/tests/test_harness_integration.py` (new)
**Changes**: End-to-end integration tests

```python
"""Integration tests for AgentHarness."""

import pytest
from agent_harness import AgentHarness, HarnessConfig, LLMConfig

@pytest.mark.asyncio
async def test_harness_basic_run():
    """Test basic harness run."""
    config = HarnessConfig(
        llm=LLMConfig(model="gpt-4o", api_key="test-key"),
        agent_name="CodeActAgent"
    )

    harness = AgentHarness(config)
    result = await harness.run("Write 'hello' to a file")
    await harness.close()

    assert result is not None
```

### Success Criteria

#### Automated Verification:
- [ ] Controller tests pass: `pytest agent-harness/tests/test_controller.py -v`
- [ ] LLM integration tests pass: `pytest agent-harness/tests/test_llm.py -v`
- [ ] Integration tests pass: `pytest agent-harness/tests/test_harness_integration.py -v`
- [ ] All tests pass: `pytest agent-harness/tests/ -v`
- [ ] Type checking passes: `mypy agent-harness/agent_harness/`
- [ ] Linting passes: `ruff check agent-harness/agent_harness/`
- [ ] No OpenHands imports: `grep -r "from openhands" agent-harness/agent_harness/` returns nothing

#### Manual Verification:
- [ ] Can create and run a simple agent task
- [ ] BAML calls work correctly
- [ ] Events are saved and can be replayed

---

## Phase 5: Library Structure & Documentation

### Overview
Finalize library structure, add documentation, and create example usage. Prepare for distribution.

### Changes Required

#### 1. Create Public API
**File**: `agent-harness/agent_harness/__init__.py` (update)
**Changes**: Export public API

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

#### 2. Create README
**File**: `agent-harness/README.md` (new)
**Changes**: Comprehensive README with usage examples

#### 3. Create Example Usage
**File**: `agent-harness/examples/simple_agent.py` (new)
**Changes**: Simple example showing basic usage

#### 4. Create Requirements File
**File**: `agent-harness/requirements.txt` (new)
**Changes**: List dependencies

```
baml-py>=0.213.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

#### 5. Create VERSION File
**File**: `agent-harness/VERSION` (new)
**Changes**: Version tracking

```
0.1.0
```

### Success Criteria

#### Automated Verification:
- [ ] README exists: `test -f agent-harness/README.md`
- [ ] Example exists: `test -f agent-harness/examples/simple_agent.py`
- [ ] Requirements file exists: `test -f agent-harness/requirements.txt`
- [ ] VERSION file exists: `test -f agent-harness/VERSION`
- [ ] Public API exports correctly: `python -c "from agent_harness import AgentHarness; print('OK')"`

#### Manual Verification:
- [ ] README is clear and comprehensive
- [ ] Example code runs successfully
- [ ] Library can be imported in a fresh project
- [ ] Documentation is helpful for new users

---

## Testing Strategy

### Unit Tests

**Component Coverage:**
- ✅ Interfaces (abstract contracts)
- ✅ Storage implementations (LocalStorage, MemoryStorage)
- ✅ Runtime implementations (LocalRuntime)
- ✅ Event system (Event, EventStream)
- ✅ State management (State, StateTracker)
- ✅ Agent base class
- ✅ LLM integration (BAML client)
- ✅ AgentController

**Test Structure:**
```
agent-harness/tests/
├── test_interfaces.py
├── test_storage_implementations.py
├── test_runtime_implementations.py
├── test_events.py
├── test_state.py
├── test_agent.py
├── test_controller.py
├── test_llm.py
└── test_harness_integration.py
```

### Integration Tests

**End-to-End Scenarios:**
1. Simple agent task execution
2. Event persistence and replay
3. State save/restore
4. BAML LLM calls
5. Custom runtime integration
6. Custom storage integration

### Manual Testing Steps

1. **Fresh Project Test:**
   ```bash
   mkdir test-project
   cd test-project
   cp -r ../agent-harness libs/agent-harness
   python -c "import sys; sys.path.insert(0, 'libs/agent-harness'); from agent_harness import AgentHarness; print('Success')"
   ```

2. **Example Execution:**
   ```bash
   cd agent-harness/examples
   python simple_agent.py
   ```

3. **BAML Generation:**
   ```bash
   cd agent-harness/agent_harness/baml_schemas
   baml update-client
   ```

## Performance Considerations

- **Event Streaming**: Use async I/O for event persistence
- **State Management**: Minimize state serialization overhead
- **BAML Calls**: Leverage BAML's built-in optimizations
- **Storage**: LocalStorage uses file system, consider caching for frequent reads

## Migration Notes

**From OpenHands to Standalone:**
- Import paths change: `openhands.*` → `agent_harness.*`
- Config models simplified: `OpenHandsConfig` → `HarnessConfig`
- Storage uses interface: `FileStore` → `StorageInterface`
- Runtime uses interface: OpenHands runtime → `RuntimeInterface`
- Logger uses standard: `openhands.core.logger` → `logging`

**Migration Guide:**
Create `agent-harness/docs/migration.md` with examples.

## References

- Original research: `thoughts/shared/research/2025-11-25-standalone-harness-library.md`
- Related research: `thoughts/shared/research/2025-11-25-encapsulated-agent-harness-baml.md`
- Agent architecture: `thoughts/shared/research/2025-11-25-agent-architecture-research.md`
- OpenHands AgentController: `openhands/controller/agent_controller.py:99`
- OpenHands EventStream: `openhands/events/stream.py:43`
- OpenHands State: `openhands/controller/state/state.py:49`


