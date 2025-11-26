---
date: 2025-11-25T20:36:38-05:00
researcher: Auto
git_commit: 2bbe15a329e35f5156cdafcbe63c3fd54978ff98
branch: main
repository: silmari-OpenHands
topic: "Encapsulated Agent Harness with BAML API Layer: Design for Cross-Project Reusability"
tags: [research, codebase, agents, architecture, baml, harness, reusable, api-layer]
status: complete
last_updated: 2025-11-25
last_updated_by: Auto
---

# Research: Encapsulated Agent Harness with BAML API Layer

**Date**: 2025-11-25T20:36:38-05:00
**Researcher**: Auto
**Git Commit**: 2bbe15a329e35f5156cdafcbe63c3fd54978ff98
**Branch**: main
**Repository**: silmari-OpenHands

## Research Question

How can we create a more fully encapsulated agent harness that can be easily used from project to project using BAML as the API layer? This research expands on the existing agent architecture research to design a reusable, project-agnostic agent system.

## Summary

This research documents the design for creating a fully encapsulated agent harness that can be reused across projects using BAML (Boundary AI Markup Language) as the API layer. The harness would abstract away OpenHands-specific implementation details while preserving the core agent capabilities.

**Key Findings:**

1. **BAML Integration**: BAML is already integrated in OpenHands, providing type-safe LLM calls with enhanced tool formatting. It will serve as the exclusive API layer for the reusable harness (no fallback mechanism).

2. **Core Components**: The agent system requires five core components: EventStream, AgentController, Runtime, LLMRegistry, and State. These can be abstracted into a harness interface.

3. **Configuration Pattern**: Configuration uses Pydantic models (OpenHandsConfig, AgentConfig, LLMConfig) that can be simplified for harness use.

4. **Tool Registration**: Tools are registered per-agent via a registry pattern, allowing custom tools to be added without modifying core code.

5. **Abstraction Points**: Key areas to abstract include runtime execution, file storage, event persistence, and LLM provider configuration.

## Detailed Findings

### 1. BAML as API Layer

#### 1.1 Current BAML Integration

BAML is integrated as an optional layer in OpenHands (`openhands/llm/llm.py:342-372`):

```342:372:openhands/llm/llm.py
            # Route to BAML if enabled
            resp: ModelResponse
            if self.use_baml and call_baml_completion is not None:
                try:
                    # Prepare kwargs for BAML
                    baml_kwargs = {
                        'temperature': kwargs.get('temperature'),
                        'max_completion_tokens': kwargs.get('max_completion_tokens') or kwargs.get('max_tokens'),
                        'max_tokens': kwargs.get('max_completion_tokens') or kwargs.get('max_tokens'),
                        'top_p': kwargs.get('top_p'),
                        'top_k': kwargs.get('top_k'),
                        'seed': kwargs.get('seed'),
                        'stop': kwargs.get('stop')
                    }

                    # Call BAML completion
                    resp = call_baml_completion(
                        messages=messages,
                        tools=kwargs.get('tools'),
                        **baml_kwargs
                    )
                    logger.debug('BAML completion successful')
                except Exception as e:
                    logger.error(f'BAML completion failed: {e}')
                    raise RuntimeError(f'BAML completion failed: {e}') from e
```

**Key Features:**
- Type-safe function definitions in BAML schema files
- Enhanced tool formatting with detailed parameter information
- Async support via `call_baml_completion_async()`
- BAML-first design with no fallback dependencies

#### 1.2 BAML Schema Structure

BAML schemas are defined in `openhands/llm/baml_src/`:

**Completion Function** (`completion.baml:20-32`):
```20:32:openhands/llm/baml_src/completion.baml
function CompleteLLMRequest(request: FormattedLLMRequest) -> LLMCompletionResponse {
  client LiteLLMClient

  prompt #"
    {{ request.conversation }}

    Please respond to the conversation above.

    {{ ctx.output_format }}
  "#
}
```

**Type Definitions** (`types.baml:1-64`):
- `FormattedLLMRequest`: Simplified request format
- `LLMCompletionResponse`: Response structure compatible with standard LLM response format
- `Message`, `TextContent`, `ImageContent`: Message structure types

**Client Configuration** (`clients.baml:4-13`):
```4:13:openhands/llm/baml_src/clients.baml
client<llm> LiteLLMClient {
  provider openai-generic
  options {
    // These will be set dynamically from LLMConfig via environment variables
    base_url env.BAML_BASE_URL?
    api_key env.BAML_API_KEY?
    model env.BAML_MODEL?
  }
}
```

#### 1.3 BAML Adapter Functions

The adapter (`openhands/llm/baml_adapter.py`) provides conversion functions:

- `convert_messages_to_baml()`: Converts OpenHands Message objects to BAML format
- `convert_tools_to_baml()`: Converts tool definitions to BAML format
- `call_baml_completion()`: Synchronous BAML completion call
- `call_baml_completion_async()`: Asynchronous BAML completion call
- `convert_baml_response_to_model_response()`: Converts BAML response to standard ModelResponse format

#### 1.4 Benefits of BAML as API Layer

1. **Type Safety**: Compile-time type checking for LLM requests and responses
2. **Schema-Driven**: Function definitions in BAML files, not Python code
3. **Enhanced Tool Formatting**: Detailed parameter information in prompts
4. **Provider Abstraction**: Client configuration abstracts LLM provider details
5. **Version Control**: BAML schemas can be versioned and shared across projects

### 2. Core Components for Harness

#### 2.1 Required Components

Based on `openhands/core/setup.py`, the minimal setup requires:

**1. EventStream** (`openhands/events/stream.py:43-150`):
- Central event bus for all actions and observations
- Subscriber pattern for components (Runtime, AgentController, Memory)
- Thread-safe event queue with async processing
- File-based persistence via FileStore

**2. AgentController** (`openhands/controller/agent_controller.py:99-202`):
- Manages agent execution loop
- Handles state management and delegation
- Subscribes to EventStream for events
- Coordinates with Runtime for action execution

**3. Runtime** (`openhands/runtime/base.py`):
- Executes actions in sandboxed environment
- Handles file operations, command execution, browser interactions
- Subscribes to EventStream for actions
- Provides workspace isolation

**4. LLMRegistry** (`openhands/llm/llm_registry.py`):
- Manages LLM instances per service
- Handles configuration and initialization
- Supports BAML integration via `use_baml` flag
- Tracks metrics and costs

**5. State** (`openhands/controller/state/state.py:48-101`):
- Tracks agent state, history, and metrics
- Manages event range filtering (start_id/end_id)
- Handles delegation state isolation
- Persists to FileStore for session restoration

#### 2.2 Initialization Flow

From `openhands/core/setup.py:212-245`:

```212:245:openhands/core/setup.py
def create_controller(
    agent: Agent,
    runtime: Runtime,
    config: OpenHandsConfig,
    conversation_stats: ConversationStats,
    headless_mode: bool = True,
    replay_events: list[Event] | None = None,
) -> tuple[AgentController, State | None]:
    event_stream = runtime.event_stream
    initial_state = None
    try:
        logger.debug(
            f'Trying to restore agent state from session {event_stream.sid} if available'
        )
        initial_state = State.restore_from_session(
            event_stream.sid, event_stream.file_store
        )
    except Exception as e:
        logger.debug(f'Cannot restore agent state: {e}')

    controller = AgentController(
        agent=agent,
        conversation_stats=conversation_stats,
        iteration_delta=config.max_iterations,
        budget_per_task_delta=config.max_budget_per_task,
        agent_to_llm_config=config.get_agent_to_llm_config_map(),
        event_stream=event_stream,
        initial_state=initial_state,
        headless_mode=headless_mode,
        confirmation_mode=config.security.confirmation_mode,
        replay_events=replay_events,
        security_analyzer=runtime.security_analyzer,
    )
    return (controller, initial_state)
```

**Initialization Order:**
1. Create EventStream with FileStore
2. Create Runtime with EventStream
3. Create LLMRegistry with config
4. Create Agent with AgentConfig and LLMRegistry
5. Create AgentController with Agent, Runtime, and EventStream

### 3. Configuration Abstraction

#### 3.1 Current Configuration Structure

Configuration uses nested Pydantic models:

**OpenHandsConfig** (`openhands/core/config/openhands_config.py:23-180`):
- `llms: dict[str, LLMConfig]` - LLM configurations
- `agents: dict[str, AgentConfig]` - Agent configurations
- `sandbox: SandboxConfig` - Runtime/sandbox settings
- `security: SecurityConfig` - Security settings
- `runtime: str` - Runtime type ('docker', 'local', 'cli')
- `file_store: str` - Storage backend type

**AgentConfig** (`openhands/core/config/agent_config.py:15-100`):
- Tool enablement flags (`enable_cmd`, `enable_editor`, `enable_jupyter`, etc.)
- `llm_config: str | None` - LLM config group name
- `condenser: CondenserConfig` - History condensation settings
- `model_routing: ModelRoutingConfig` - Model routing settings
- `runtime: str | None` - Runtime type override

**LLMConfig** (`openhands/core/config/llm_config.py:12`):
- `use_baml: bool = False` - Enable BAML integration
- `model: str` - Model identifier
- `api_key: SecretStr | None` - API key
- `base_url: str | None` - Base URL for API
- Temperature, max_tokens, and other LLM parameters

#### 3.2 Simplified Harness Configuration

For a reusable harness, configuration can be simplified:

```python
@dataclass
class HarnessConfig:
    """Simplified configuration for agent harness."""
    # LLM Configuration
    llm_provider: str  # 'openai', 'anthropic', 'ollama', etc.
    llm_model: str
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    use_baml: bool = True  # Default to BAML

    # Agent Configuration
    agent_name: str = 'CodeActAgent'
    tools: list[str] = field(default_factory=lambda: ['cmd', 'editor', 'think'])

    # Runtime Configuration
    runtime_type: str = 'docker'  # 'docker', 'local', 'cli'
    workspace_path: str = './workspace'

    # Storage Configuration
    storage_type: str = 'local'  # 'local', 's3', 'memory'
    storage_path: str = '~/.agent-harness'

    # Execution Configuration
    max_iterations: int = 100
    max_budget: float | None = None
    headless: bool = True
```

### 4. Tool Registration Pattern

#### 4.1 Current Tool System

Tools are registered per-agent in `_get_tools()` method:

```108:153:openhands/agenthub/codeact_agent/codeact_agent.py
    def _get_tools(self) -> list['ChatCompletionToolParam']:
        # For these models, we use short tool descriptions ( < 1024 tokens)
        # to avoid hitting the OpenAI token limit for tool descriptions.
        SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS = ['gpt-4', 'o3', 'o1', 'o4']

        use_short_tool_desc = False
        if self.llm is not None:
            # For historical reasons, previously OpenAI enforces max function description length of 1k characters
            # https://community.openai.com/t/function-call-description-max-length/529902
            # But it no longer seems to be an issue recently
            # https://community.openai.com/t/was-the-character-limit-for-schema-descriptions-upgraded/1225975
            # Tested on GPT-5 and longer description still works. But we still keep the logic to be safe for older models.
            use_short_tool_desc = any(
                model_substr in self.llm.config.model
                for model_substr in SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS
            )

        tools = []
        if self.config.enable_cmd:
            tools.append(create_cmd_run_tool(use_short_description=use_short_tool_desc))
        if self.config.enable_think:
            tools.append(ThinkTool)
        if self.config.enable_finish:
            tools.append(FinishTool)
        if self.config.enable_condensation_request:
            tools.append(CondensationRequestTool)
        if self.config.enable_browsing:
            if sys.platform == 'win32':
                logger.warning('Windows runtime does not support browsing yet')
            else:
                tools.append(BrowserTool)
        if self.config.enable_jupyter:
            tools.append(IPythonTool)
        if self.config.enable_plan_mode:
            # In plan mode, we use the task_tracker tool for task management
            tools.append(create_task_tracker_tool(use_short_tool_desc))
        if self.config.enable_llm_editor:
            tools.append(LLMBasedFileEditTool)
        elif self.config.enable_editor:
            tools.append(
                create_str_replace_editor_tool(
                    use_short_description=use_short_tool_desc,
                    runtime_type=self.config.runtime,
                )
            )
        return tools
```

#### 4.2 Harness Tool Registration

For a reusable harness, tools can be registered via:

1. **Configuration-based**: Tools specified in config (as shown above)
2. **Plugin system**: Tools loaded from external modules
3. **BAML tool definitions**: Tools defined in BAML schemas for type safety

**Example BAML Tool Definition:**
```baml
// Tool definition in BAML schema
class FileReadTool {
  name string
  description string
  parameters {
    path string @description("Path to file to read")
    start_line int? @description("Starting line number (optional)")
    end_line int? @description("Ending line number (optional)")
  }
}

function ReadFile(tool: FileReadTool) -> FileContent {
  // Tool implementation
}
```

### 5. Abstraction Points for Reusability

#### 5.1 Runtime Abstraction

The Runtime interface (`openhands/runtime/base.py`) can be abstracted:

```python
class RuntimeInterface(ABC):
    """Abstract interface for runtime execution."""

    @abstractmethod
    async def execute_action(self, action: Action) -> Observation:
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
```

**Implementation Options:**
- Docker-based runtime (current OpenHands implementation)
- Local process runtime (for development)
- Remote runtime (for distributed execution)
- Custom runtime (project-specific implementations)

#### 5.2 Storage Abstraction

FileStore interface (`openhands/storage/`) can be abstracted:

```python
class StorageInterface(ABC):
    """Abstract interface for event and file storage."""

    @abstractmethod
    async def save_event(self, event: Event) -> None:
        """Save event to storage."""
        pass

    @abstractmethod
    async def load_events(self, session_id: str) -> list[Event]:
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
```

**Implementation Options:**
- Local file system (current default)
- S3/object storage (for cloud deployments)
- Database storage (for structured queries)
- In-memory storage (for testing)

#### 5.3 LLM Provider Abstraction

BAML already provides LLM provider abstraction via client configuration. The harness can expose this:

```python
class LLMProviderInterface(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        **kwargs
    ) -> ModelResponse:
        """Complete LLM request."""
        pass
```

**BAML Implementation:**
- Configure BAML clients for different providers
- Use BAML function definitions for type safety
- BAML is required - no fallback mechanism

#### 5.4 Event Stream Abstraction

EventStream can be abstracted for different backends:

```python
class EventStreamInterface(ABC):
    """Abstract interface for event streaming."""

    @abstractmethod
    def add_event(self, event: Event, source: EventSource) -> None:
        """Add event to stream."""
        pass

    @abstractmethod
    def subscribe(
        self,
        subscriber_id: str,
        callback: Callable[[Event], None],
        callback_id: str
    ) -> None:
        """Subscribe to events."""
        pass

    @abstractmethod
    def search_events(
        self,
        start_id: int = 0,
        end_id: int | None = None
    ) -> list[Event]:
        """Search events by ID range."""
        pass
```

### 6. Harness Architecture Design

#### 6.1 Proposed Harness Structure

```
agent-harness/
├── core/
│   ├── harness.py           # Main harness class
│   ├── agent.py             # Agent interface
│   ├── controller.py        # AgentController abstraction
│   └── state.py             # State management
├── runtime/
│   ├── interface.py         # RuntimeInterface
│   ├── docker_runtime.py    # Docker implementation
│   └── local_runtime.py     # Local implementation
├── storage/
│   ├── interface.py         # StorageInterface
│   ├── local_storage.py     # Local file system
│   └── s3_storage.py        # S3 implementation
├── llm/
│   ├── baml_client.py       # BAML client wrapper
│   ├── provider.py          # LLMProviderInterface
│   └── registry.py         # LLM registry
├── events/
│   ├── stream.py            # EventStreamInterface
│   └── event.py             # Event definitions
├── tools/
│   ├── registry.py          # Tool registry
│   └── base.py              # Tool interface
├── config/
│   ├── harness_config.py    # HarnessConfig
│   └── baml_schemas/        # BAML schema files
│       ├── completion.baml
│       ├── types.baml
│       └── clients.baml
└── examples/
    └── simple_agent.py      # Usage example
```

#### 6.2 Harness API Design

**Main Harness Class:**
```python
class AgentHarness:
    """Encapsulated agent harness for cross-project reuse."""

    def __init__(self, config: HarnessConfig):
        """Initialize harness with configuration."""
        self.config = config
        self.event_stream = self._create_event_stream()
        self.runtime = self._create_runtime()
        self.llm_registry = self._create_llm_registry()
        self.agent = self._create_agent()
        self.controller = self._create_controller()

    async def run(self, task: str) -> str:
        """Run agent with given task."""
        # Add initial user message
        # Start controller loop
        # Wait for completion
        # Return result

    async def step(self) -> None:
        """Execute one agent step."""
        await self.controller.step()

    def get_state(self) -> State:
        """Get current agent state."""
        return self.controller.state

    async def close(self) -> None:
        """Clean up harness resources."""
        await self.controller.close()
        await self.runtime.teardown()
        self.event_stream.close()
```

#### 6.3 BAML Integration in Harness

**BAML Configuration:**
```python
class BAMLHarnessConfig(HarnessConfig):
    """Harness config with BAML-specific settings."""
    baml_schema_path: str = './baml_schemas'
    baml_client_path: str = './baml_client'
    use_baml: bool = True

    def setup_baml(self) -> None:
        """Setup BAML environment variables."""
        os.environ['BAML_API_KEY'] = self.llm_api_key or ''
        os.environ['BAML_BASE_URL'] = self.llm_base_url or ''
        os.environ['BAML_MODEL'] = self.llm_model
```

**BAML Tool Definitions:**
Tools can be defined in BAML schemas for type safety:

```baml
// tools.baml
class FileReadAction {
  path string
  start_line int?
  end_line int?
}

class FileEditAction {
  path string
  edits EditCommand[]
}

function ExecuteFileRead(action: FileReadAction) -> FileContent {
  client LiteLLMClient
  // Implementation
}

function ExecuteFileEdit(action: FileEditAction) -> EditResult {
  client LiteLLMClient
  // Implementation
}
```

### 7. Usage Example

#### 7.1 Simple Usage

```python
from agent_harness import AgentHarness, HarnessConfig

# Configure harness
config = HarnessConfig(
    llm_provider='openai',
    llm_model='gpt-4o',
    llm_api_key='sk-...',
    use_baml=True,
    agent_name='CodeActAgent',
    tools=['cmd', 'editor', 'think'],
    runtime_type='docker',
    workspace_path='./workspace',
    max_iterations=50
)

# Create and run harness
harness = AgentHarness(config)
result = await harness.run("Write a Python script to calculate fibonacci numbers")
print(result)

# Clean up
await harness.close()
```

#### 7.2 Advanced Usage with Custom Tools

```python
from agent_harness import AgentHarness, HarnessConfig, ToolRegistry

# Register custom tool
@ToolRegistry.register('custom_tool')
class CustomTool:
    def __init__(self):
        self.name = 'custom_tool'
        self.description = 'A custom tool for project-specific tasks'

    def get_schema(self) -> dict:
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'param1': {'type': 'string'}
                    }
                }
            }
        }

# Use custom tool
config = HarnessConfig(
    llm_provider='openai',
    llm_model='gpt-4o',
    use_baml=True,
    tools=['cmd', 'editor', 'custom_tool']  # Include custom tool
)

harness = AgentHarness(config)
result = await harness.run("Use custom_tool to do something")
```

#### 7.3 BAML-First Usage

```python
from agent_harness import BAMLHarnessConfig, AgentHarness

# BAML-specific configuration
config = BAMLHarnessConfig(
    llm_provider='openai',
    llm_model='gpt-4o',
    llm_api_key='sk-...',
    use_baml=True,
    baml_schema_path='./my_project/baml_schemas',
    baml_client_path='./my_project/baml_client'
)

# Setup BAML environment
config.setup_baml()

# Create harness (will use BAML for all LLM calls)
harness = AgentHarness(config)
result = await harness.run("Task description")
```

### 8. Migration Path

#### 8.1 From OpenHands to Harness

**Step 1: Extract Core Components**
- Copy EventStream, AgentController, State classes
- Extract Runtime interface and implementations
- Extract LLMRegistry and BAML integration

**Step 2: Create Abstraction Layer**
- Define interfaces for Runtime, Storage, EventStream
- Create factory functions for component creation
- Implement configuration simplification

**Step 3: BAML Schema Migration**
- Move BAML schemas to harness package
- Update client generation paths
- Ensure BAML functions match harness needs

**Step 4: Tool System Migration**
- Extract tool definitions
- Create tool registry system
- Support BAML tool definitions

**Step 5: Testing and Validation**
- Create test suite for harness
- Validate BAML integration
- Test cross-project compatibility

### 9. Benefits of BAML-Based Harness

#### 9.1 Type Safety

BAML provides compile-time type checking:
- Function signatures are validated at build time
- Tool parameters are type-checked
- Response types are guaranteed

#### 9.2 Schema-Driven Development

- LLM interactions defined in BAML files, not Python
- Schemas can be versioned and shared
- Changes to LLM interface don't require Python code changes

#### 9.3 Enhanced Tool Formatting

BAML automatically formats tools with detailed parameter information:
- Parameter names, types, and descriptions
- Required vs optional parameters
- Nested object structures

#### 9.4 Provider Abstraction

BAML client configuration abstracts LLM provider details:
- Switch providers by changing client config
- Support multiple providers simultaneously
- Easy integration of new providers

#### 9.5 Cross-Project Reusability

- BAML schemas can be shared across projects
- Tool definitions in BAML are portable
- Configuration is standardized

## Code References

### BAML Integration
- `openhands/llm/llm.py:342-372` - BAML routing in LLM completion
- `openhands/llm/baml_adapter.py:335-416` - BAML adapter functions
- `openhands/llm/baml_src/completion.baml:20-32` - BAML completion function
- `openhands/llm/baml_src/types.baml:1-64` - BAML type definitions
- `openhands/llm/baml_src/clients.baml:4-13` - BAML client configuration

### Core Components
- `openhands/core/setup.py:35-245` - Component initialization functions
- `openhands/controller/agent_controller.py:99-202` - AgentController initialization
- `openhands/events/stream.py:43-150` - EventStream implementation
- `openhands/controller/state/state.py:48-101` - State class definition

### Configuration
- `openhands/core/config/openhands_config.py:23-180` - OpenHandsConfig
- `openhands/core/config/agent_config.py:15-100` - AgentConfig
- `openhands/core/config/llm_config.py:12` - LLMConfig with use_baml flag

### Tool System
- `openhands/agenthub/codeact_agent/codeact_agent.py:108-153` - Tool registration
- `openhands/agenthub/codeact_agent/tools/__init__.py:1-19` - Tool exports

## Architecture Documentation

### Current Architecture

The OpenHands agent system uses:
1. **Event-Driven Architecture**: All actions and observations flow through EventStream
2. **Subscriber Pattern**: Components subscribe to EventStream for events
3. **State Isolation**: Each agent has separate State with event range filtering
4. **Delegation Pattern**: Parent agents can spawn delegate agents for subtasks
5. **Runtime Abstraction**: Runtime interface allows different execution environments

### Proposed Harness Architecture

The encapsulated harness would:
1. **Abstract Core Components**: Provide interfaces for Runtime, Storage, EventStream
2. **BAML-First Design**: Use BAML as the exclusive API layer for LLM interactions (no fallback)
3. **Configuration Simplification**: Reduce configuration complexity for common use cases
4. **Tool Registry**: Allow dynamic tool registration without code changes
5. **Cross-Project Portability**: Enable reuse across different projects

### Component Dependencies

```
AgentHarness
├── EventStreamInterface
│   └── StorageInterface
├── RuntimeInterface
│   └── EventStreamInterface
├── LLMRegistry
│   └── BAMLClient (required, no fallback)
├── Agent
│   ├── LLMRegistry
│   └── ToolRegistry
└── AgentController
    ├── Agent
    ├── EventStreamInterface
    ├── RuntimeInterface
    └── State
```

## Historical Context (from thoughts/)

- `thoughts/shared/research/2025-11-25-agent-architecture-research.md` - Original agent architecture research covering file system handling, tool calls, looping strategy, and context windows
- `thoughts/shared/research/2025-11-16-baml-llm-integration.md` - Research on configuring LLM calls to use BAML
- `thoughts/shared/plans/2025-11-16-baml-llm-integration.md` - Implementation plan for BAML LLM integration

## Related Research

- `thoughts/shared/research/2025-11-25-agent-architecture-research.md` - Foundation research on agent architecture

## Open Questions

1. **BAML Schema Versioning**: How should BAML schemas be versioned across projects?
2. **Tool Compatibility**: How to ensure tools work across different runtime implementations?
3. **State Persistence**: What's the best approach for state persistence in a reusable harness?
4. **Error Handling**: How should errors be handled and reported in a harness context?
5. **Performance**: What performance characteristics does BAML provide compared to direct API calls?
6. **Testing**: How to test harness with different BAML configurations?
7. **Deployment**: What's the best way to package and distribute the harness?
8. **Backward Compatibility**: How to maintain compatibility with existing OpenHands code?

## Recommendations

### Immediate Next Steps

1. **Create Harness Prototype**: Build minimal harness with BAML integration
2. **Define Interfaces**: Create abstract interfaces for Runtime, Storage, EventStream
3. **BAML Schema Design**: Design BAML schemas for harness-specific functions
4. **Tool Registry**: Implement dynamic tool registration system
5. **Configuration Simplification**: Create simplified config model

### Long-Term Considerations

1. **Schema Sharing**: Develop mechanism for sharing BAML schemas across projects
2. **Plugin System**: Design plugin system for custom tools and runtimes
3. **Documentation**: Create comprehensive documentation for harness usage
4. **Examples**: Build example projects demonstrating harness usage
5. **Testing Framework**: Develop testing framework for harness validation

