---
date: 2025-11-16T17:12:29-05:00
researcher: Auto
git_commit: 2bbe15a329e35f5156cdafcbe63c3fd54978ff98
branch: main
repository: silmari-OpenHands
topic: "BAML LLM Integration Implementation Plan"
tags: [plan, implementation, llm, baml, integration]
status: draft
last_updated: 2025-11-16
last_updated_by: Auto
---

# BAML LLM Integration Implementation Plan

## Overview

This plan outlines the integration of BAML (Boundary AI Markup Language) into OpenHands' LLM call system. The integration will make BAML an optional layer that can be enabled via configuration, allowing gradual adoption while maintaining full backward compatibility with the existing LiteLLM-based implementation.

## Current State Analysis

### Existing Architecture

**Core Components:**
- `openhands/llm/llm.py` - Main `LLM` class wrapping `litellm.completion()`
- `openhands/llm/llm_registry.py` - Manages LLM instances per service
- `openhands/core/config/llm_config.py` - Configuration via `LLMConfig` Pydantic model
- `openhands/core/message.py` - `Message` class for structured message handling
- `openhands/llm/fn_call_converter.py` - Function calling conversion logic

**Current Flow:**
1. Agent gets LLM from `LLMRegistry`
2. Agent calls `self.llm.completion(messages=..., tools=...)`
3. `LLM.completion()` formats messages, handles function calling, calls `litellm.completion()`
4. Returns `ModelResponse` from LiteLLM
5. Agent processes response via `response_to_actions()`

**Key Discoveries:**
- BAML has been initialized at `openhands/llm/baml_src/` with example files
- BAML client generated but currently configured for TypeScript (needs Python)
- Current implementation uses `litellm.types.utils.ModelResponse` as return type
- Messages are converted from `Message` objects to dict format via `format_messages_for_llm()`
- Function calling has extensive conversion logic for models that don't support it natively
- Supports async (`AsyncLLM`) and streaming (`StreamingLLM`) modes

### BAML Setup Status

**Current State:**
- ✅ BAML source directory exists: `openhands/llm/baml_src/`
- ✅ Example BAML files present: `clients.baml`, `resume.baml`, `generators.baml`
- ⚠️ Generator configured for TypeScript, needs Python: `output_type "typescript"` → should be `"python/pydantic"`
- ✅ BAML client directory exists: `openhands/llm/baml_client/` (TypeScript files)

**What's Missing:**
- Python BAML client generation
- BAML function definitions for LLM completion
- Integration code to use BAML instead of direct LiteLLM calls
- Configuration flag to enable/disable BAML
- Response format conversion from BAML to `ModelResponse`

## Desired End State

After this implementation:

1. **Configuration-Driven**: Users can enable BAML via `use_baml = true` in `LLMConfig` or TOML config
2. **Backward Compatible**: When BAML is disabled (default), existing LiteLLM behavior is unchanged
3. **Feature Parity**: BAML integration supports all existing features:
   - Function calling (native and mocked)
   - Streaming responses
   - Async completion
   - Vision support
   - Prompt caching
   - Cost tracking
   - Metrics and logging
4. **Type Safety**: BAML functions use structured schemas matching OpenHands' message format
5. **Verification**: Automated tests verify BAML integration works identically to LiteLLM

### Success Criteria

#### Automated Verification:
- [ ] BAML client generates successfully: `baml update-client` from `openhands/llm/baml_src/`
- [ ] Unit tests pass for BAML integration: `pytest tests/unit/llm/test_baml_llm.py`
- [ ] Integration tests pass: `pytest tests/integration/test_baml_agent.py`
- [ ] Type checking passes: `mypy openhands/llm/baml_llm.py`
- [ ] Linting passes: `ruff check openhands/llm/baml_llm.py`
- [ ] Existing LLM tests still pass (backward compatibility): `pytest tests/unit/llm/test_llm.py`
- [ ] Function calling tests pass with BAML: `pytest tests/unit/llm/test_function_calling.py`

#### Manual Verification:
- [ ] Agent works correctly with BAML enabled via config
- [ ] Function calling works identically with BAML vs LiteLLM
- [ ] Streaming responses work with BAML
- [ ] Async completion works with BAML
- [ ] Cost tracking and metrics work correctly
- [ ] Performance is acceptable (no significant overhead)
- [ ] Error handling works correctly with BAML

## What We're NOT Doing

| Out of Scope | What We ARE Doing |
|-------------|-------------------|
| Replacing LiteLLM entirely | Making BAML an optional layer on top |
| Removing existing LiteLLM code | Adding BAML as alternative path |
| Changing agent interfaces | Maintaining same `LLM.completion()` API |
| Modifying message format | Using existing `Message` class |
| Changing function calling logic | Reusing existing conversion functions |
| Breaking backward compatibility | Default behavior unchanged (BAML disabled) |
| Supporting BAML-only mode | Always supporting LiteLLM fallback |

## Implementation Approach

**Strategy: Optional BAML Wrapper (Option 3 from research)**

We'll implement BAML as an optional layer that can be enabled via configuration. The approach:

1. **Add configuration flag** to `LLMConfig`: `use_baml: bool = False`
2. **Create BAML function definitions** that match the LLM completion interface
3. **Create BAML adapter** in `LLM.completion()` that routes to BAML when enabled
4. **Convert BAML responses** to `ModelResponse` format for compatibility
5. **Maintain all existing features** (function calling, streaming, async, etc.)

This approach:
- ✅ Maintains backward compatibility (default: BAML disabled)
- ✅ Allows gradual adoption
- ✅ Preserves all existing functionality
- ✅ Minimal changes to existing code
- ✅ Easy to test and verify

## Phase 1: BAML Configuration and Client Setup

### Overview
Configure BAML to generate Python client code and set up the basic infrastructure for BAML integration.

### Changes Required:

#### 1. Update BAML Generator Configuration
**File**: `openhands/llm/baml_src/generators.baml`
**Changes**: Change output type from TypeScript to Python

```baml
generator target {
    // Change from "typescript" to "python/pydantic"
    output_type "python/pydantic"

    // Where the generated code will be saved (relative to baml_src/)
    output_dir "../baml_client"

    // The version of the BAML package you have installed
    version "0.213.0"

    // Use sync mode for compatibility with existing LLM.completion() interface
    default_client_mode sync
}
```

#### 2. Add BAML Dependency
**File**: `pyproject.toml`
**Changes**: Add baml-py dependency

```toml
[tool.poetry.dependencies]
# ... existing dependencies ...
baml-py = "^0.213.0"
```

#### 3. Add use_baml Configuration Flag
**File**: `openhands/core/config/llm_config.py`
**Changes**: Add `use_baml` field to `LLMConfig`

```python
class LLMConfig(BaseModel):
    # ... existing fields ...

    use_baml: bool = Field(
        default=False,
        description="Whether to use BAML for LLM calls instead of direct LiteLLM calls"
    )
```

#### 4. Update Config Template
**File**: `config.template.toml`
**Changes**: Add use_baml option to LLM config section

```toml
[llm]
# ... existing options ...
# Whether to use BAML for LLM calls instead of direct LiteLLM calls
#use_baml = false
```

### Success Criteria:

#### Automated Verification:
- [x] BAML Python client generates: `cd openhands/llm/baml_src && baml update-client`
- [x] Python client files exist: `ls openhands/llm/baml_client/*.py`
- [x] Type checking passes: `mypy openhands/core/config/llm_config.py`
- [x] Config validation works: `python -c "from openhands.core.config import LLMConfig; LLMConfig(use_baml=True)"`

#### Manual Verification:
- [ ] BAML client can be imported: `from openhands.llm.baml_client.sync_client import b`
- [ ] Config file accepts use_baml option without errors

---

## Phase 2: BAML Function Definitions

### Overview
Create BAML function definitions that match the LLM completion interface, accepting messages, tools, and configuration parameters.

### Changes Required:

#### 1. Create BAML Types for Messages
**File**: `openhands/llm/baml_src/types.baml`
**Changes**: Define BAML classes matching OpenHands message structure

```baml
// Message content types
class TextContent {
  type string
  text string
  cache_control map<string, string>?
}

class ImageContent {
  type string
  image_url map<string, string>
  cache_control map<string, string>?
}

// Union type for content
type Content = TextContent | ImageContent

// Message class matching OpenHands Message structure
class Message {
  role "user" | "system" | "assistant" | "tool"
  content Content[]
  tool_calls map<string, any>[]?
  tool_call_id string?
  name string?
}

// Tool definition for function calling
class Tool {
  type string
  function map<string, any>
}

// LLM completion request
class LLMCompletionRequest {
  messages Message[]
  tools Tool[]?
  temperature float?
  max_tokens int?
  top_p float?
  top_k float?
  seed int?
  stop string[]?
  // Add other parameters as needed
}

// Response structure matching ModelResponse
class Choice {
  index int
  message Message
  finish_reason string?
}

class Usage {
  prompt_tokens int?
  completion_tokens int?
  total_tokens int?
  prompt_tokens_details map<string, any>?
}

class LLMCompletionResponse {
  id string
  choices Choice[]
  usage Usage?
  model string?
  created int?
}
```

#### 2. Create BAML Client Configuration
**File**: `openhands/llm/baml_src/clients.baml`
**Changes**: Add client configuration that uses LiteLLM provider or direct providers

```baml
// Client that uses LiteLLM (for compatibility)
client<llm> LiteLLMClient {
  provider openai-generic
  options {
    // These will be set dynamically from LLMConfig
    base_url env.BAML_BASE_URL?
    api_key env.BAML_API_KEY?
    model env.BAML_MODEL?
  }
}

// Alternative: Direct provider clients (commented for now)
// client<llm> AnthropicClient {
//   provider anthropic
//   options {
//     model "claude-sonnet-4-20250514"
//     api_key env.ANTHROPIC_API_KEY
//   }
// }
```

#### 3. Create BAML Completion Function
**File**: `openhands/llm/baml_src/completion.baml`
**Changes**: Define BAML function for LLM completion

```baml
import types
import clients

// Main completion function
function CompleteLLMRequest(request: types.LLMCompletionRequest) -> types.LLMCompletionResponse {
  // Use LiteLLM client or configure based on model
  client clients.LiteLLMClient

  // For now, we'll use a simple prompt structure
  // The actual messages will be passed via the request parameter
  // This is a placeholder - actual implementation will need to handle
  // the message format conversion
  prompt #"
    {{ ctx.messages }}
  "#

  // Note: BAML may need custom handling for function calling and tools
  // This will be refined in Phase 3
}
```

**Note**: This is a simplified version. The actual implementation will need to handle:
- Message format conversion
- Function calling/tools
- Different providers based on model name
- All the parameters from LLMConfig

### Success Criteria:

#### Automated Verification:
- [x] BAML files compile: `baml validate` from `openhands/llm/baml_src/`
- [x] BAML client regenerates with new functions: `baml update-client`
- [x] Python client has new function: `from openhands.llm.baml_client.sync_client import b; hasattr(b, 'CompleteLLMRequest')`

#### Manual Verification:
- [ ] BAML function can be called (even if not fully functional yet)
- [ ] Types match expected structure

---

## Phase 3: BAML Adapter Implementation

### Overview
Create the adapter code that integrates BAML into the existing `LLM` class, routing calls to BAML when enabled.

### Changes Required:

#### 1. Create BAML Adapter Module
**File**: `openhands/llm/baml_adapter.py` (new file)
**Changes**: Create adapter to convert between OpenHands and BAML formats

```python
"""Adapter for BAML LLM integration.

This module provides functions to convert between OpenHands' message format
and BAML's format, and to convert BAML responses back to ModelResponse format.
"""

from typing import Any

from litellm.types.utils import ModelResponse

try:
    from openhands.llm.baml_client.sync_client import b
    from openhands.llm.baml_client import types as baml_types
except ImportError:
    # BAML client not available
    b = None
    baml_types = None

from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message


def convert_messages_to_baml(
    messages: list[dict[str, Any]] | list[Message],
) -> list[baml_types.Message]:
    """Convert OpenHands messages to BAML Message format.

    Args:
        messages: List of messages in OpenHands format (dict or Message objects)

    Returns:
        List of BAML Message objects
    """
    if baml_types is None:
        raise ImportError("BAML client not available. Run 'baml update-client'.")

    baml_messages = []
    for msg in messages:
        if isinstance(msg, Message):
            # Convert Message object to dict first
            msg_dict = msg.model_dump()
        else:
            msg_dict = msg

        # Convert to BAML format
        baml_content = []
        if isinstance(msg_dict.get('content'), str):
            # String content
            baml_content.append(
                baml_types.TextContent(
                    type='text',
                    text=msg_dict['content']
                )
            )
        elif isinstance(msg_dict.get('content'), list):
            # List content (for vision, etc.)
            for content_item in msg_dict['content']:
                if content_item.get('type') == 'text':
                    baml_content.append(
                        baml_types.TextContent(
                            type='text',
                            text=content_item.get('text', '')
                        )
                    )
                elif content_item.get('type') == 'image_url':
                    baml_content.append(
                        baml_types.ImageContent(
                            type='image_url',
                            image_url={'url': content_item['image_url']['url']}
                        )
                    )

        baml_msg = baml_types.Message(
            role=msg_dict['role'],
            content=baml_content,
            tool_calls=msg_dict.get('tool_calls'),
            tool_call_id=msg_dict.get('tool_call_id'),
            name=msg_dict.get('name')
        )
        baml_messages.append(baml_msg)

    return baml_messages


def convert_tools_to_baml(
    tools: list[dict[str, Any]] | None,
) -> list[baml_types.Tool] | None:
    """Convert OpenHands tools to BAML Tool format.

    Args:
        tools: List of tools in OpenHands format

    Returns:
        List of BAML Tool objects or None
    """
    if tools is None or baml_types is None:
        return None

    baml_tools = []
    for tool in tools:
        baml_tools.append(
            baml_types.Tool(
                type=tool.get('type', 'function'),
                function=tool.get('function', {})
            )
        )
    return baml_tools


def convert_baml_response_to_model_response(
    baml_response: baml_types.LLMCompletionResponse,
) -> ModelResponse:
    """Convert BAML response to LiteLLM ModelResponse format.

    Args:
        baml_response: BAML completion response

    Returns:
        ModelResponse compatible with LiteLLM format
    """
    # Convert BAML response to dict format matching ModelResponse
    choices = []
    for choice in baml_response.choices:
        # Convert BAML message back to dict
        message_dict = {
            'role': choice.message.role,
            'content': _convert_baml_content_to_dict(choice.message.content)
        }
        if choice.message.tool_calls:
            message_dict['tool_calls'] = choice.message.tool_calls
        if choice.message.tool_call_id:
            message_dict['tool_call_id'] = choice.message.tool_call_id
        if choice.message.name:
            message_dict['name'] = choice.message.name

        choices.append({
            'index': choice.index,
            'message': message_dict,
            'finish_reason': choice.finish_reason
        })

    usage_dict = None
    if baml_response.usage:
        usage_dict = {
            'prompt_tokens': baml_response.usage.prompt_tokens,
            'completion_tokens': baml_response.usage.completion_tokens,
            'total_tokens': baml_response.usage.total_tokens,
            'prompt_tokens_details': baml_response.usage.prompt_tokens_details
        }

    return ModelResponse(
        id=baml_response.id,
        choices=choices,
        usage=usage_dict,
        model=baml_response.model,
        created=baml_response.created
    )


def _convert_baml_content_to_dict(
    content: list[baml_types.TextContent | baml_types.ImageContent]
) -> str | list[dict[str, Any]]:
    """Convert BAML content to OpenHands format.

    Returns either a string (for simple text) or a list of content blocks.
    """
    if len(content) == 1 and isinstance(content[0], baml_types.TextContent):
        # Simple text content
        return content[0].text

    # Multiple content blocks (vision, etc.)
    result = []
    for item in content:
        if isinstance(item, baml_types.TextContent):
            result.append({
                'type': 'text',
                'text': item.text
            })
        elif isinstance(item, baml_types.ImageContent):
            result.append({
                'type': 'image_url',
                'image_url': item.image_url
            })
    return result


def call_baml_completion(
    messages: list[dict[str, Any]] | list[Message],
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> ModelResponse:
    """Call BAML completion function and return ModelResponse.

    Args:
        messages: List of messages
        tools: Optional list of tools for function calling
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        ModelResponse compatible with LiteLLM format
    """
    if b is None:
        raise ImportError("BAML client not available. Run 'baml update-client'.")

    # Convert messages and tools to BAML format
    baml_messages = convert_messages_to_baml(messages)
    baml_tools = convert_tools_to_baml(tools)

    # Create BAML request
    request = baml_types.LLMCompletionRequest(
        messages=baml_messages,
        tools=baml_tools,
        temperature=kwargs.get('temperature'),
        max_tokens=kwargs.get('max_tokens') or kwargs.get('max_completion_tokens'),
        top_p=kwargs.get('top_p'),
        top_k=kwargs.get('top_k'),
        seed=kwargs.get('seed'),
        stop=kwargs.get('stop')
    )

    # Call BAML function
    baml_response = b.CompleteLLMRequest(request)

    # Convert back to ModelResponse
    return convert_baml_response_to_model_response(baml_response)
```

#### 2. Integrate BAML into LLM Class
**File**: `openhands/llm/llm.py`
**Changes**: Add BAML routing in `__init__` and `completion` wrapper

```python
# Add import at top
from openhands.llm.baml_adapter import call_baml_completion

class LLM(RetryMixin, DebugMixin):
    def __init__(
        self,
        config: LLMConfig,
        service_id: str,
        metrics: Metrics | None = None,
        retry_listener: Callable[[int, int], None] | None = None,
    ) -> None:
        # ... existing initialization code ...

        # Store BAML flag
        self.use_baml = config.use_baml

        # If using BAML, set up environment variables for BAML client
        if self.use_baml:
            if self.config.api_key:
                os.environ['BAML_API_KEY'] = self.config.api_key.get_secret_value()
            if self.config.base_url:
                os.environ['BAML_BASE_URL'] = self.config.base_url
            if self.config.model:
                os.environ['BAML_MODEL'] = self.config.model

        # ... rest of existing initialization ...

        # Modify the wrapper function to route to BAML when enabled
        @self.retry_decorator(
            num_retries=self.config.num_retries,
            retry_exceptions=LLM_RETRY_EXCEPTIONS,
            retry_min_wait=self.config.retry_min_wait,
            retry_max_wait=self.config.retry_max_wait,
            retry_multiplier=self.config.retry_multiplier,
            retry_listener=self.retry_listener,
        )
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for the litellm completion function. Logs the input and output of the completion function."""
            from openhands.io import json

            # ... existing message processing code (lines 226-301) ...

            # Route to BAML if enabled
            if self.use_baml:
                try:
                    # Prepare kwargs for BAML
                    baml_kwargs = {
                        'temperature': kwargs.get('temperature'),
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

                    # Continue with existing response processing (lines 333-408)
                    # ... rest of wrapper code ...
                except Exception as e:
                    logger.warning(f"BAML completion failed, falling back to LiteLLM: {e}")
                    # Fall back to LiteLLM
                    self.use_baml = False
                    # Continue with existing LiteLLM code

            # ... existing LiteLLM completion code (lines 331-408) ...
```

**Note**: This is a simplified version. The actual implementation needs to:
- Handle all the message formatting logic before routing
- Maintain function calling conversion logic
- Handle all edge cases and error scenarios
- Preserve all existing behavior

### Success Criteria:

#### Automated Verification:
- [x] BAML adapter imports successfully: `from openhands.llm.baml_adapter import call_baml_completion`
- [ ] Unit tests pass: `pytest tests/unit/llm/test_baml_adapter.py`
- [ ] LLM class works with BAML enabled: `pytest tests/unit/llm/test_llm.py -k baml`
- [x] Type checking passes: `mypy openhands/llm/baml_adapter.py openhands/llm/llm.py`

#### Manual Verification:
- [ ] LLM completion works with `use_baml = true` in config
- [ ] Response format matches expected ModelResponse structure
- [ ] Error handling works (falls back to LiteLLM on BAML failure)

---

## Phase 4: Function Calling and Advanced Features

### Overview
Ensure BAML integration supports all advanced features: function calling, streaming, async, vision, and prompt caching.

### Changes Required:

#### 1. Enhance BAML Function for Function Calling
**File**: `openhands/llm/baml_src/completion.baml`
**Changes**: Update to handle tools and function calling properly

```baml
import types
import clients

function CompleteLLMRequest(request: types.LLMCompletionRequest) -> types.LLMCompletionResponse {
  client clients.LiteLLMClient

  // BAML needs to support tools parameter
  // This may require using BAML's tool calling features
  // or passing tools as part of the prompt/messages

  // For now, we'll need to investigate BAML's function calling support
  // and adapt accordingly

  prompt #"
    // Messages will be formatted here
    // Tools will need to be included if present
  "#
}
```

**Note**: This phase requires research into BAML's function calling capabilities. We may need to:
- Use BAML's native tool calling if supported
- Or convert tools to prompt format (similar to existing mock function calling)

#### 2. Add Async BAML Support
**File**: `openhands/llm/async_llm.py`
**Changes**: Add BAML routing for async completion

```python
# Similar to sync LLM, but use async BAML client
from openhands.llm.baml_client.async_client import b as async_b

# Add async BAML completion function in baml_adapter.py
async def call_baml_completion_async(
    messages: list[dict[str, Any]] | list[Message],
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> ModelResponse:
    """Async version of BAML completion."""
    # Similar to sync version but using async_b
    # ...
```

#### 3. Add Streaming BAML Support
**File**: `openhands/llm/streaming_llm.py`
**Changes**: Add BAML routing for streaming (if BAML supports it)

**Note**: Streaming support depends on BAML's capabilities. May need to:
- Use BAML streaming if available
- Or fall back to LiteLLM for streaming

#### 4. Handle Vision and Prompt Caching
**File**: `openhands/llm/baml_adapter.py`
**Changes**: Ensure message conversion handles vision content and caching flags

```python
def convert_messages_to_baml(
    messages: list[dict[str, Any]] | list[Message],
) -> list[baml_types.Message]:
    # ... existing code ...

    # Handle vision content (already in ImageContent conversion)
    # Handle cache_control flags if present in messages
    # ...
```

### Success Criteria:

#### Automated Verification:
- [ ] Function calling tests pass with BAML: `pytest tests/unit/llm/test_function_calling.py -k baml`
- [ ] Async tests pass: `pytest tests/unit/llm/test_async_llm.py -k baml`
- [ ] Vision tests pass: `pytest tests/unit/llm/test_vision.py -k baml`
- [ ] Integration tests pass: `pytest tests/integration/test_baml_agent.py`

#### Manual Verification:
- [ ] Function calling works identically with BAML vs LiteLLM
- [ ] Streaming works (if supported by BAML)
- [ ] Async completion works
- [ ] Vision support works
- [ ] Prompt caching works (if supported by BAML)

---

## Phase 5: Testing and Documentation

### Overview
Create comprehensive tests and documentation for BAML integration.

### Changes Required:

#### 1. Unit Tests for BAML Adapter
**File**: `tests/unit/llm/test_baml_adapter.py` (new file)
**Changes**: Test message conversion, response conversion, and error handling

```python
"""Tests for BAML adapter."""

import pytest
from litellm.types.utils import ModelResponse

from openhands.core.message import Message, TextContent
from openhands.llm.baml_adapter import (
    convert_baml_response_to_model_response,
    convert_messages_to_baml,
    convert_tools_to_baml,
)


def test_convert_messages_to_baml():
    """Test message conversion to BAML format."""
    messages = [
        Message(role='user', content=[TextContent(text='Hello')])
    ]
    baml_messages = convert_messages_to_baml(messages)
    assert len(baml_messages) == 1
    assert baml_messages[0].role == 'user'
    # ... more assertions ...


def test_convert_baml_response_to_model_response():
    """Test BAML response conversion to ModelResponse."""
    # Create mock BAML response
    # Convert and verify structure
    # ...


# Add more test cases for edge cases, error handling, etc.
```

#### 2. Integration Tests
**File**: `tests/integration/test_baml_agent.py` (new file)
**Changes**: Test agent behavior with BAML enabled

```python
"""Integration tests for BAML with agents."""

import pytest

from openhands.core.config import AgentConfig, LLMConfig, OpenHandsConfig
from openhands.llm.llm_registry import LLMRegistry


def test_codeact_agent_with_baml():
    """Test CodeActAgent works with BAML enabled."""
    config = OpenHandsConfig()
    config.llm_configs['llm'] = LLMConfig(
        model='gpt-4o',
        use_baml=True
    )

    registry = LLMRegistry(config)
    agent = CodeActAgent(config.agent_configs['agent'], registry)

    # Test agent can make LLM calls
    # ...
```

#### 3. Update Documentation
**File**: `openhands/llm/README.md` or create new doc
**Changes**: Document BAML integration

```markdown
# BAML Integration

BAML (Boundary AI Markup Language) can be used as an alternative to direct LiteLLM calls.

## Configuration

Enable BAML in your config:

```toml
[llm]
use_baml = true
model = "gpt-4o"
api_key = "your-key"
```

## Usage

BAML is automatically used when `use_baml = true` is set in LLMConfig.
No code changes are required - the same `LLM.completion()` interface is used.

## Features

- ✅ Function calling
- ✅ Async completion
- ⚠️ Streaming (depends on BAML support)
- ✅ Vision support
- ⚠️ Prompt caching (depends on BAML support)

## Troubleshooting

If BAML fails, the system automatically falls back to LiteLLM.
Check logs for BAML-related errors.
```

### Success Criteria:

#### Automated Verification:
- [ ] All unit tests pass: `pytest tests/unit/llm/test_baml_adapter.py -v`
- [ ] All integration tests pass: `pytest tests/integration/test_baml_agent.py -v`
- [ ] Test coverage meets threshold: `pytest --cov=openhands/llm/baml_adapter --cov-report=term-missing`
- [ ] Documentation builds without errors

#### Manual Verification:
- [ ] Documentation is clear and complete
- [ ] Examples work as documented
- [ ] Troubleshooting guide is helpful

---

## Testing Strategy

### Unit Tests:
- **Message Conversion**: Test `convert_messages_to_baml()` with various message formats
- **Response Conversion**: Test `convert_baml_response_to_model_response()` with various response structures
- **Tool Conversion**: Test `convert_tools_to_baml()` with function calling tools
- **Error Handling**: Test fallback behavior when BAML fails
- **Edge Cases**: Empty messages, missing fields, invalid formats

### Integration Tests:
- **Agent Integration**: Test CodeActAgent with BAML enabled
- **Function Calling**: Test function calling works end-to-end with BAML
- **Async Completion**: Test async LLM calls with BAML
- **Configuration**: Test BAML can be enabled/disabled via config

### Manual Testing Steps:
1. Enable BAML in config: `use_baml = true`
2. Run a simple agent task
3. Verify LLM calls are made via BAML (check logs)
4. Test function calling with tools
5. Test async completion
6. Test error scenarios (invalid API key, network failure)
7. Verify fallback to LiteLLM works

## Performance Considerations

- **Overhead**: BAML adds a conversion layer. Measure performance impact.
- **Caching**: Consider caching BAML client instances if possible.
- **Error Handling**: Fast fallback to LiteLLM on BAML errors.

## Migration Notes

- **Backward Compatibility**: Default `use_baml = false` ensures no breaking changes
- **Gradual Adoption**: Users can enable BAML per LLM config
- **Rollback**: Simply set `use_baml = false` to revert to LiteLLM

## References

- Original research: `thoughts/shared/research/2025-11-16-baml-llm-integration.md`
- BAML documentation: https://docs.boundaryml.com
- LiteLLM documentation: https://docs.litellm.ai
- Current LLM implementation: `openhands/llm/llm.py:54-419`
- Message format: `openhands/core/message.py:53-159`

