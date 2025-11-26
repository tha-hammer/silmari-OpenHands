"""Adapter for BAML LLM integration.

This module provides functions to convert between harness message format
and BAML's format, and to convert BAML responses back to ModelResponse format.
"""

from typing import Any, Optional

try:
    from agent_harness.baml_client.baml_client.sync_client import b  # type: ignore
    from agent_harness.baml_client.baml_client.async_client import b as async_b  # type: ignore
    from agent_harness.baml_client.baml_client import types as baml_types  # type: ignore
except ImportError:
    # BAML client not available
    b: Optional[Any] = None  # type: ignore[assignment]
    async_b: Optional[Any] = None  # type: ignore[assignment]
    baml_types: Optional[Any] = None  # type: ignore[assignment]

from agent_harness.utils.logging import setup_logger

logger = setup_logger(__name__)


def _format_messages_to_text(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Format messages and tools to a text conversation format.

    This converts structured messages into a text format that BAML can process.

    Args:
        messages: List of messages
        tools: Optional list of tools for function calling

    Returns:
        Formatted conversation text
    """
    conversation_parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Format content based on type
        if isinstance(content, str):
            content_text = content
        elif isinstance(content, list):
            content_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        content_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {})
                        if isinstance(image_url, dict):
                            content_parts.append(f"[Image: {image_url.get('url', '')}]")
                else:
                    content_parts.append(str(item))
            content_text = " ".join(content_parts)
        else:
            content_text = str(content)

        # Format message by role
        if role == "system":
            conversation_parts.append(f"System: {content_text}")
        elif role == "user":
            conversation_parts.append(f"User: {content_text}")
        elif role == "assistant":
            conversation_parts.append(f"Assistant: {content_text}")
        elif role == "tool":
            name = msg.get("name", "tool")
            conversation_parts.append(f"Tool ({name}): {content_text}")

        # Add tool calls if present
        if msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    func_name = tool_call.get("function", {}).get("name", "unknown")
                    func_args = tool_call.get("function", {}).get("arguments", "{}")
                    conversation_parts.append(
                        f"Assistant calls function: {func_name}({func_args})"
                    )

    # Add tools information if provided
    if tools:
        tools_text = "Available tools:\n"
        for tool in tools:
            if isinstance(tool, dict):
                func_info = tool.get("function", {})
                func_name = func_info.get("name", "unknown")
                func_desc = func_info.get("description", "")
                func_params = func_info.get("parameters", {})

                # Format function signature with parameters
                tools_text += f"\n{func_name}: {func_desc}\n"

                # Add parameter details if available
                if func_params and isinstance(func_params, dict):
                    properties = func_params.get("properties", {})
                    required = func_params.get("required", [])
                    if properties:
                        tools_text += "  Parameters:\n"
                        for param_name, param_info in properties.items():
                            param_type = param_info.get("type", "string")
                            param_desc = param_info.get("description", "")
                            is_required = param_name in required
                            req_marker = " (required)" if is_required else " (optional)"
                            tools_text += (
                                f"    - {param_name} ({param_type}){req_marker}: {param_desc}\n"
                            )
        conversation_parts.append(tools_text)

    return "\n\n".join(conversation_parts)


def convert_baml_response_to_model_response(
    baml_response: Any,
) -> dict[str, Any]:
    """Convert BAML response to ModelResponse format.

    Args:
        baml_response: BAML completion response (LLMCompletionResponse)

    Returns:
        Dict compatible with LiteLLM ModelResponse format
    """
    if baml_types is None:
        raise ImportError("BAML client not available. Run 'baml update-client'.")

    # BAML returns LLMCompletionResponse with:
    # - content: str
    # - tool_calls: Optional[Dict[str, str]]
    # - prompt_tokens, completion_tokens, total_tokens: Optional[int]

    # Convert to ModelResponse format with choices array
    message_dict: dict[str, Any] = {
        "role": "assistant",
        "content": baml_response.content,
    }

    # Add tool_calls if present
    if baml_response.tool_calls:
        # Convert dict format to list of tool calls
        tool_calls_list = []
        for call_id, call_data in baml_response.tool_calls.items():
            # Parse call_data if it's a string (JSON)
            import json

            try:
                if isinstance(call_data, str):
                    call_dict = json.loads(call_data)
                else:
                    call_dict = call_data
            except (json.JSONDecodeError, TypeError):
                call_dict = {"name": call_id, "arguments": str(call_data)}

            tool_calls_list.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call_dict.get("name", call_id),
                        "arguments": json.dumps(call_dict.get("arguments", call_dict)),
                    },
                }
            )
        message_dict["tool_calls"] = tool_calls_list

    usage_dict = None
    if (
        baml_response.prompt_tokens is not None
        or baml_response.completion_tokens is not None
    ):
        usage_dict = {
            "prompt_tokens": baml_response.prompt_tokens or 0,
            "completion_tokens": baml_response.completion_tokens or 0,
            "total_tokens": baml_response.total_tokens
            or (baml_response.prompt_tokens or 0) + (baml_response.completion_tokens or 0),
        }

    return {
        "id": f"baml-{id(baml_response)}",
        "choices": [
            {
                "index": 0,
                "message": message_dict,
                "finish_reason": "stop",
            }
        ],
        "usage": usage_dict,
        "model": None,  # Model info not in BAML response
        "created": None,  # Timestamp not in BAML response
    }


def call_baml_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Call BAML completion function and return ModelResponse.

    Args:
        messages: List of messages
        tools: Optional list of tools for function calling
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        Dict compatible with LiteLLM ModelResponse format

    Raises:
        ImportError: If BAML client is not available
    """
    if b is None:
        raise ImportError("BAML client not available. Run 'baml update-client'.")

    # Format messages to text for BAML
    conversation_text = _format_messages_to_text(messages, tools)

    # Create BAML request with formatted conversation
    request = baml_types.FormattedLLMRequest(
        conversation=conversation_text,
        temperature=kwargs.get("temperature"),
        max_tokens=kwargs.get("max_tokens") or kwargs.get("max_completion_tokens"),
        top_p=kwargs.get("top_p"),
        top_k=kwargs.get("top_k"),
        seed=kwargs.get("seed"),
        stop=kwargs.get("stop"),
    )

    # Call BAML function
    baml_response = b.CompleteLLMRequest(request)

    # Convert back to ModelResponse
    return convert_baml_response_to_model_response(baml_response)


async def call_baml_completion_async(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Call BAML completion function asynchronously and return ModelResponse.

    Args:
        messages: List of messages
        tools: Optional list of tools for function calling
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        Dict compatible with LiteLLM ModelResponse format

    Raises:
        ImportError: If BAML async client is not available
    """
    if async_b is None:
        raise ImportError("BAML async client not available. Run 'baml update-client'.")

    # Format messages to text for BAML
    conversation_text = _format_messages_to_text(messages, tools)

    # Create BAML request with formatted conversation
    request = baml_types.FormattedLLMRequest(
        conversation=conversation_text,
        temperature=kwargs.get("temperature"),
        max_tokens=kwargs.get("max_tokens") or kwargs.get("max_completion_tokens"),
        top_p=kwargs.get("top_p"),
        top_k=kwargs.get("top_k"),
        seed=kwargs.get("seed"),
        stop=kwargs.get("stop"),
    )

    # Call BAML function asynchronously
    baml_response = await async_b.CompleteLLMRequest(request)

    # Convert back to ModelResponse
    return convert_baml_response_to_model_response(baml_response)

