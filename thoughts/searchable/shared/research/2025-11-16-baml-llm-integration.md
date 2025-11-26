---
date: 2025-11-16T17:12:29-05:00
researcher: Auto
git_commit: 2bbe15a329e35f5156cdafcbe63c3fd54978ff98
branch: main
repository: silmari-OpenHands
topic: "Configuring LLM calls to use BAML"
tags: [research, codebase, llm, baml, integration]
status: complete
last_updated: 2025-11-16
last_updated_by: Auto
---

# Research: Configuring LLM calls to use BAML

**Date**: 2025-11-16T17:12:29-05:00
**Researcher**: Auto
**Git Commit**: 2bbe15a329e35f5156cdafcbe63c3fd54978ff98
**Branch**: main
**Repository**: silmari-OpenHands

## Research Question

How can LLM calls in OpenHands be configured to use BAML (Boundary AI Markup Language) instead of direct LiteLLM calls?

## Summary

OpenHands currently uses **LiteLLM** as the underlying library for all LLM interactions. The system architecture centers around:

1. **LLM Class** (`openhands/llm/llm.py`) - Wraps `litellm.completion()` with retry logic, logging, and cost tracking
2. **LLMRegistry** (`openhands/llm/llm_registry.py`) - Manages LLM instances per service
3. **Agent Integration** - Agents get LLM instances from the registry and call `self.llm.completion()` with messages and tools

**BAML** is a language for defining LLM functions with structured schemas. It generates client code that can be imported and used in Python. To integrate BAML, we would need to:

- Create a BAML adapter/wrapper that implements the same interface as the current `LLM.completion()` method
- Define BAML functions that accept messages and tools as parameters
- Map OpenHands' message format to BAML function calls
- Potentially replace or wrap the `litellm.completion()` calls with BAML client calls

## Detailed Findings

### Current LLM Architecture

#### Core LLM Implementation

The main LLM class is located at `openhands/llm/llm.py`:

```54:419:openhands/llm/llm.py
class LLM(RetryMixin, DebugMixin):
    """The LLM class represents a Language Model instance.

    Attributes:
        config: an LLMConfig object specifying the configuration of the LLM.
    """

    def __init__(
        self,
        config: LLMConfig,
        service_id: str,
        metrics: Metrics | None = None,
        retry_listener: Callable[[int, int], None] | None = None,
    ) -> None:
        """Initializes the LLM. If LLMConfig is passed, its values will be the fallback.

        Passing simple parameters always overrides config.

        Args:
            config: The LLM configuration.
            metrics: The metrics to use.
        """
        # ... initialization code ...

        # set up the completion function
        kwargs: dict[str, Any] = {
            'temperature': self.config.temperature,
            'max_completion_tokens': self.config.max_output_tokens,
        }
        # ... more kwargs setup ...

        self._completion = partial(
            litellm_completion,
            model=self.config.model,
            api_key=self.config.api_key.get_secret_value()
            if self.config.api_key
            else None,
            base_url=self.config.base_url,
            api_version=self.config.api_version,
            custom_llm_provider=self.config.custom_llm_provider,
            timeout=self.config.timeout,
            drop_params=self.config.drop_params,
            seed=self.config.seed,
            **kwargs,
        )

        # ... wrapper function with retry logic, logging, etc. ...

    @property
    def completion(self) -> Callable:
        """Decorator for the litellm completion function.

        Check the complete documentation at https://litellm.vercel.app/docs/completion
        """
        return self._completion
```

Key points:
- Uses `litellm.completion` as the underlying function
- Wraps it with retry logic, logging, cost tracking, and message formatting
- Handles function calling conversion for models that don't support it natively
- Supports streaming, async, and sync modes

#### LLM Registry

The registry manages LLM instances:

```23:147:openhands/llm/llm_registry.py
class LLMRegistry:
    def __init__(
        self,
        config: OpenHandsConfig,
        agent_cls: str | None = None,
        retry_listener: Callable[[int, int], None] | None = None,
    ):
        self.registry_id = str(uuid4())
        self.config = copy.deepcopy(config)
        self.retry_listner = retry_listener
        self.agent_to_llm_config = self.config.get_agent_to_llm_config_map()
        self.service_to_llm: dict[str, LLM] = {}
        self.subscriber: Callable[[Any], None] | None = None

        selected_agent_cls = self.config.default_agent
        if agent_cls:
            selected_agent_cls = agent_cls

        agent_name = selected_agent_cls if selected_agent_cls is not None else 'agent'
        llm_config = self.config.get_llm_config_from_agent(agent_name)
        self.active_agent_llm: LLM = self.get_llm('agent', llm_config)

    def _create_new_llm(
        self, service_id: str, config: LLMConfig, with_listener: bool = True
    ) -> LLM:
        if with_listener:
            llm = LLM(
                service_id=service_id, config=config, retry_listener=self.retry_listner
            )
        else:
            llm = LLM(service_id=service_id, config=config)
        self.service_to_llm[service_id] = llm
        self.notify(RegistryEvent(llm=llm, service_id=service_id))
        return llm

    def get_llm(
        self,
        service_id: str,
        config: LLMConfig | None = None,
    ):
        logger.info(
            f'[LLM registry {self.registry_id}]: Registering service for {service_id}'
        )

        # Attempting to switch configs for existing LLM
        if (
            service_id in self.service_to_llm
            and self.service_to_llm[service_id].config != config
        ):
            raise ValueError(
                f'Requesting same service ID {service_id} with different config, use a new service ID'
            )

        if service_id in self.service_to_llm:
            return self.service_to_llm[service_id]

        if not config:
            raise ValueError('Requesting new LLM without specifying LLM config')

        return self._create_new_llm(config=config, service_id=service_id)
```

#### Agent Usage Pattern

Agents get LLM instances from the registry and call completion:

```78:225:openhands/agenthub/codeact_agent/codeact_agent.py
    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        """Initializes a new instance of the CodeActAgent class.

        Parameters:
        - config (AgentConfig): The configuration for this agent
        """
        super().__init__(config, llm_registry)
        # ... initialization ...

        # Override with router if needed
        self.llm = self.llm_registry.get_router(self.config)

    # ... later in the code ...

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message)
        params: dict = {
            'messages': messages,
        }
        params['tools'] = check_tools(self.tools, self.llm.config)
        params['extra_body'] = {
            'metadata': state.to_llm_metadata(
                model_name=self.llm.config.model, agent_name=self.name
            )
        }
        response = self.llm.completion(**params)
        logger.debug(f'Response from LLM: {response}')
        actions = self.response_to_actions(response)
```

### LLM Configuration

Configuration is handled through `LLMConfig`:

```12:195:openhands/core/config/llm_config.py
class LLMConfig(BaseModel):
    """Configuration for the LLM model.

    Attributes:
        model: The model to use.
        api_key: The API key to use.
        base_url: The base URL for the API. This is necessary for local LLMs.
        api_version: The version of the API.
        aws_access_key_id: The AWS access key ID.
        aws_secret_access_key: The AWS secret access key.
        aws_region_name: The AWS region name.
        num_retries: The number of retries to attempt.
        retry_multiplier: The multiplier for the exponential backoff.
        retry_min_wait: The minimum time to wait between retries, in seconds. This is exponential backoff minimum. For models with very low limits, this can be set to 15-20.
        retry_max_wait: The maximum time to wait between retries, in seconds. This is exponential backoff maximum.
        timeout: The timeout for the API.
        max_message_chars: The approximate max number of characters in the content of an event included in the prompt to the LLM. Larger observations are truncated.
        temperature: The temperature for the API.
        top_p: The top p for the API.
        top_k: The top k for the API.
        custom_llm_provider: The custom LLM provider to use. This is undocumented in openhands, and normally not used. It is documented on the litellm side.
        max_input_tokens: The maximum number of input tokens. Note that this is currently unused, and the value at runtime is actually the total tokens in OpenAI (e.g. 128,000 tokens for GPT-4).
        max_output_tokens: The maximum number of output tokens. This is sent to the LLM.
        input_cost_per_token: The cost per input token. This will available in logs for the user to check.
        output_cost_per_token: The cost per output token. This will available in logs for the user to check.
        ollama_base_url: The base URL for the OLLAMA API.
        drop_params: Drop any unmapped (unsupported) params without causing an exception.
        modify_params: Modify params allows litellm to do transformations like adding a default message, when a message is empty.
        disable_vision: If model is vision capable, this option allows to disable image processing (useful for cost reduction).
        caching_prompt: Use the prompt caching feature if provided by the LLM and supported by the provider.
        log_completions: Whether to log LLM completions to the state.
        log_completions_folder: The folder to log LLM completions to. Required if log_completions is True.
        custom_tokenizer: A custom tokenizer to use for token counting.
        native_tool_calling: Whether to use native tool calling if supported by the model. Can be True, False, or not set.
        reasoning_effort: The effort to put into reasoning. This is a string that can be one of 'low', 'medium', 'high', or 'none'. Can apply to all reasoning models.
        seed: The seed to use for the LLM.
        safety_settings: Safety settings for models that support them (like Mistral AI and Gemini).
        for_routing: Whether this LLM is used for routing. This is set to True for models used in conjunction with the main LLM in the model routing feature.
    """

    model: str = Field(default='claude-sonnet-4-20250514')
    api_key: SecretStr | None = Field(default=None)
    base_url: str | None = Field(default=None)
    # ... more fields ...
```

Configuration is loaded from TOML files (see `config.template.toml`).

### BAML Overview

BAML (Boundary AI Markup Language) is a language for defining LLM functions with structured schemas. Key characteristics:

1. **Function Definition**: BAML functions are defined in `.baml` files with input/output schemas
2. **Client Generation**: Running `baml update-client` generates Python client code in `baml_client/`
3. **Usage Pattern**: Import and call functions like `b.FunctionName()` from the generated client
4. **Provider Support**: BAML supports LiteLLM as a provider, which means it can use LiteLLM under the hood

From BAML documentation:
- BAML functions are called via `from baml_client.sync_client import b` then `b.FunctionName()`
- BAML supports LiteLLM provider configuration
- BAML generates type-safe client code

### Integration Points

#### Primary Integration Point: LLM.completion()

The main integration point would be the `LLM.completion()` method. Currently it:

1. Formats messages for the LLM
2. Handles function calling conversion
3. Calls `litellm.completion()`
4. Processes the response
5. Logs and tracks metrics

To use BAML, we could:

**Option 1: Replace litellm.completion() with BAML client calls**
- Define BAML functions that accept messages and tools
- Call BAML functions instead of `litellm.completion()`
- Map BAML responses back to the expected format

**Option 2: Create a BAML adapter class**
- Create a `BAMLLLM` class that implements the same interface as `LLM`
- Use BAML client internally but maintain the same API
- This would allow gradual migration

**Option 3: Make BAML optional via configuration**
- Add a `use_baml` flag to `LLMConfig`
- When enabled, use BAML client; otherwise use direct LiteLLM
- This provides backward compatibility

#### Message Format Conversion

OpenHands uses a `Message` class that gets converted to dict format:

```809:828:openhands/llm/llm.py
    def format_messages_for_llm(self, messages: Message | list[Message]) -> list[dict]:
        if isinstance(messages, Message):
            messages = [messages]

        # set flags to know how to serialize the messages
        for message in messages:
            message.cache_enabled = self.is_caching_prompt_active()
            message.vision_enabled = self.vision_is_active()
            message.function_calling_enabled = self.is_function_calling_active()
            # ... more flags ...

        # let pydantic handle the serialization
        return [message.model_dump() for message in messages]
```

BAML functions would need to accept this message format. The BAML function signature would need to match what OpenHands expects.

#### Function Calling Support

OpenHands has extensive function calling support:

```478:619:openhands/llm/fn_call_converter.py
def convert_fncall_messages_to_non_fncall_messages(
    messages: list[dict],
    tools: list[ChatCompletionToolParam],
    add_in_context_learning_example: bool = True,
) -> list[dict]:
    """Convert function calling messages to non-function calling messages."""
    # ... conversion logic ...
```

BAML would need to support tools/function calling in its function definitions.

### Implementation Considerations

#### 1. BAML Function Definition

A BAML function for OpenHands would need to:
- Accept messages (list of message dicts)
- Accept tools (list of tool definitions)
- Accept configuration (temperature, max_tokens, etc.)
- Return a response in the format expected by OpenHands

Example BAML function structure (conceptual):
```baml
class Message {
  role string
  content string | ContentBlock[]
}

class LLMRequest {
  messages Message[]
  tools Tool[]?
  temperature float?
  max_tokens int?
  // ... other params
}

class LLMResponse {
  id string
  choices Choice[]
  usage Usage?
}

function CompleteLLMRequest(request: LLMRequest) -> LLMResponse {
  client "openai-generic" // or litellm provider
  prompt #"
    // This would need to be configured to use the messages and tools
  "#
}
```

#### 2. Client Configuration

BAML supports LiteLLM as a provider. From BAML documentation:
```baml
client<llm> MyClient {
  provider "openai-generic"
  options {
    base_url "http://0.0.0.0:4000"
    api_key env.LITELLM_API_KEY
    model "gpt-4o"
  }
}
```

This means BAML can use LiteLLM, which is what OpenHands already uses. However, BAML would be an additional layer on top.

#### 3. Response Format Compatibility

OpenHands expects responses in LiteLLM's `ModelResponse` format. BAML responses would need to be converted to this format, or BAML functions would need to return data in a compatible structure.

#### 4. Async and Streaming Support

OpenHands has `AsyncLLM` and `StreamingLLM` classes. BAML supports async clients (`from baml_client.async_client import b`), so async support is possible. Streaming would need to be verified.

## Code References

### Core LLM Implementation
- `openhands/llm/llm.py:54-419` - Main LLM class with completion method
- `openhands/llm/llm_registry.py:23-147` - LLM registry management
- `openhands/core/config/llm_config.py:12-195` - LLM configuration

### Agent Integration
- `openhands/agenthub/codeact_agent/codeact_agent.py:78-225` - Agent using LLM completion
- `openhands/llm/fn_call_converter.py:478-619` - Function calling conversion

### Configuration
- `config.template.toml:96-228` - LLM configuration template
- `pyproject.toml:29` - LiteLLM dependency

## Architecture Documentation

### Current Flow

1. **Initialization**: `LLMRegistry` creates `LLM` instances from `LLMConfig`
2. **Agent Setup**: Agents get LLM from registry via `llm_registry.get_router(config)`
3. **Message Preparation**: Agent calls `_get_messages()` to format conversation history
4. **Completion Call**: Agent calls `self.llm.completion(messages=..., tools=...)`
5. **LLM Processing**: `LLM.completion()`:
   - Formats messages via `format_messages_for_llm()`
   - Converts function calling if needed
   - Calls `litellm.completion()`
   - Processes response, logs, tracks metrics
6. **Response Handling**: Agent processes response via `response_to_actions()`

### Integration Architecture Options

**Option A: BAML Wrapper**
```
Agent → LLM.completion() → BAML Client → LiteLLM → LLM Provider
```

**Option B: BAML Adapter**
```
Agent → BAMLLLM.completion() → BAML Client → LiteLLM → LLM Provider
```

**Option C: Direct BAML**
```
Agent → BAML Function → LLM Provider (bypassing LiteLLM)
```

Option A maintains the current interface but adds BAML as a layer. Option B creates a separate BAML-based LLM class. Option C would require more significant changes.

## Historical Context (from thoughts/)

No existing research documents found on BAML integration in the thoughts/ directory.

## Related Research

No related research documents found.

## Open Questions

1. **Performance Impact**: What is the performance overhead of adding BAML as a layer?
2. **Type Safety**: How would BAML's type system integrate with OpenHands' existing message types?
3. **Tool/Function Calling**: Does BAML support the same function calling format that OpenHands uses?
4. **Streaming**: Does BAML support streaming responses that OpenHands requires?
5. **Configuration**: How would BAML client configuration integrate with OpenHands' existing config system?
6. **Backward Compatibility**: Can BAML and direct LiteLLM calls coexist?
7. **Error Handling**: How do BAML errors map to OpenHands' error handling?
8. **Cost Tracking**: How would cost tracking work with BAML (does it expose the same metrics)?
9. **Retry Logic**: Would BAML's retry logic conflict with OpenHands' retry logic?
10. **Testing**: How would existing tests need to be modified to support BAML?

## Recommendations for Next Steps

1. **Proof of Concept**: Create a simple BAML function that accepts messages and returns a response compatible with OpenHands' expected format
2. **Interface Compatibility**: Verify that BAML can produce responses in the `ModelResponse` format that OpenHands expects
3. **Function Calling**: Test if BAML supports the tool/function calling format used by OpenHands
4. **Configuration Integration**: Design how BAML client configuration would integrate with `LLMConfig`
5. **Performance Testing**: Measure the overhead of using BAML vs direct LiteLLM calls
6. **Gradual Migration Path**: Design a way to make BAML optional so it can be tested alongside existing implementation

