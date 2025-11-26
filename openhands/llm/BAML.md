# BAML LLM Integration

## Overview

OpenHands supports optional integration with [BAML (Boundary AI Markup Language)](https://docs.boundaryml.com/) as an alternative to direct LiteLLM calls. BAML provides a type-safe, schema-driven approach to LLM interactions with enhanced tool parameter formatting and structured response handling.

## Features

- **Type-safe LLM calls**: BAML provides compile-time type checking for LLM requests and responses
- **Enhanced tool formatting**: Detailed parameter information is included in function calling prompts
- **Automatic fallback**: Falls back to LiteLLM if BAML is unavailable or encounters errors
- **Async support**: Full support for asynchronous LLM completion calls
- **Seamless integration**: Works with existing OpenHands agents and tools without code changes

## Configuration

### Enabling BAML

BAML integration is controlled by the `use_baml` flag in `LLMConfig`:

```toml
[llm]
use_baml = true
model = "gpt-4o"
api_key = "your-api-key"
base_url = "https://api.openai.com/v1"
```

Or via environment variable:

```bash
export LLM_USE_BAML=true
```

### BAML Environment Variables

When BAML is enabled, the following environment variables are automatically set from `LLMConfig`:

- `BAML_API_KEY`: Set from `LLMConfig.api_key` (empty string for local LLMs like Ollama)
- `BAML_BASE_URL`: Set from `LLMConfig.base_url`
- `BAML_MODEL`: Set from `LLMConfig.model`

These are configured automatically when the LLM is initialized, so you typically don't need to set them manually.

### Using with Local LLMs (Ollama)

BAML works seamlessly with local LLMs like Ollama:

```toml
[llm]
use_baml = true
model = "ollama/llama2"
base_url = "http://localhost:11434"
# api_key not required for Ollama
```

## How It Works

### Message Conversion

BAML integration converts OpenHands `Message` objects to BAML's format:

- Text content is converted to `TextContent`
- Image content is converted to `ImageContent`
- Tool calls and function calls are preserved
- Cache control settings are maintained

### Tool Formatting

When tools are provided, BAML formats them with detailed parameter information:

- Function names and descriptions
- Parameter names, types, and descriptions
- Required vs optional parameters
- Nested object structures

This enhanced formatting helps LLMs better understand available tools and their usage.

### Response Conversion

BAML responses are converted back to LiteLLM's `ModelResponse` format, ensuring compatibility with existing OpenHands code.

## Fallback Behavior

If BAML encounters an error (e.g., BAML client not available, configuration issue), it automatically falls back to LiteLLM:

1. BAML attempt fails
2. Warning is logged
3. `use_baml` flag is temporarily disabled for the request
4. Request is processed via LiteLLM
5. Response is returned normally

This ensures reliability even if BAML is misconfigured.

## Async Support

BAML fully supports asynchronous completion calls:

```python
from openhands.llm.async_llm import AsyncLLM

async_llm = AsyncLLM(config=llm_config, service_id='test')
response = await async_llm.async_completion(messages=messages)
```

The async implementation:
- Converts `Message` objects to dicts (for LiteLLM compatibility)
- Routes to BAML if enabled
- Falls back to LiteLLM async calls if needed

## BAML Client Setup

The BAML client is generated from BAML schema files in `openhands/llm/baml_src/`. To regenerate the client:

```bash
baml update-client
```

This generates Python client code in `openhands/llm/baml_client/`.

## Testing

Integration tests for BAML are located in `tests/integration/test_baml_agent.py`. These tests:

- Verify BAML initialization and configuration
- Test completion calls with and without BAML
- Test function calling with BAML
- Test async completion calls
- Use local Ollama LLM for real API calls (no mocking)

To run the tests:

```bash
# Ensure Ollama is running
pytest tests/integration/test_baml_agent.py -v
```

Unit tests for the BAML adapter are in `tests/unit/llm/test_baml_adapter.py`.

## Limitations

- **Streaming**: Streaming responses are not yet supported via BAML (falls back to LiteLLM)
- **Vision**: Vision capabilities work but use text-based formatting
- **Caching**: BAML's caching features are not yet fully integrated

## Troubleshooting

### BAML Client Not Available

If you see `ImportError: BAML client not available`, regenerate the client:

```bash
baml update-client
```

### Environment Variables Not Set

BAML environment variables are set automatically from `LLMConfig`. If you're seeing errors about missing variables:

1. Ensure `use_baml = true` in your config
2. Verify `model`, `base_url`, and `api_key` (if required) are set in `LLMConfig`
3. Check that the LLM instance is initialized after config is loaded

### Fallback to LiteLLM

If BAML is enabled but requests are falling back to LiteLLM, check the logs for warnings:

```
BAML completion failed, falling back to LiteLLM: <error message>
```

Common causes:
- BAML client not generated
- Missing environment variables
- BAML schema errors

## Architecture

The BAML integration consists of:

- **`baml_adapter.py`**: Core adapter functions for message/tool conversion and BAML calls
- **`baml_src/`**: BAML schema definitions
- **`baml_client/`**: Generated Python client code
- **`llm.py`**: Sync LLM routing to BAML
- **`async_llm.py`**: Async LLM routing to BAML

## Future Enhancements

Planned improvements:

- Streaming response support
- Enhanced vision support
- BAML caching integration
- Performance optimizations

## References

- [BAML Documentation](https://docs.boundaryml.com/)
- [LiteLLM Documentation](https://docs.litellm.ai/)

