"""Adapter for BAML LLM integration.

This module provides functions to convert between OpenHands' message format
and BAML's format, and to convert BAML responses back to ModelResponse format.
"""

from typing import Any

from litellm.types.utils import ModelResponse

try:
    from openhands.llm.baml_client.sync_client import b
    from openhands.llm.baml_client.async_client import b as async_b
    from openhands.llm.baml_client import types as baml_types
except ImportError:
    # BAML client not available
    b = None
    async_b = None
    baml_types = None

from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message


def convert_messages_to_baml(
    messages: list[dict[str, Any]] | list[Message],
) -> list[Any]:
    """Convert OpenHands messages to BAML Message format.

    Args:
        messages: List of messages in OpenHands format (dict or Message objects)

    Returns:
        List of BAML Message objects

    Raises:
        ImportError: If BAML client is not available
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
        content = msg_dict.get('content')

        if isinstance(content, str):
            # String content (simple text)
            baml_content.append(
                baml_types.TextContent(
                    type='text',
                    text=content
                )
            )
        elif isinstance(content, list):
            # List content (for vision, tool calls, etc.)
            for content_item in content:
                if isinstance(content_item, dict):
                    if content_item.get('type') == 'text':
                        text_content = baml_types.TextContent(
                            type='text',
                            text=content_item.get('text', '')
                        )
                        # Handle cache_control if present
                        if 'cache_control' in content_item:
                            text_content.cache_control = content_item['cache_control']
                        baml_content.append(text_content)
                    elif content_item.get('type') == 'image_url':
                        image_url = content_item.get('image_url', {})
                        if isinstance(image_url, dict):
                            image_content = baml_types.ImageContent(
                                type='image_url',
                                image_url=image_url
                            )
                            # Handle cache_control if present
                            if 'cache_control' in content_item:
                                image_content.cache_control = content_item['cache_control']
                            baml_content.append(image_content)

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
) -> list[Any] | None:
    """Convert OpenHands tools to BAML Tool format.

    Args:
        tools: List of tools in OpenHands format

    Returns:
        List of BAML Tool objects or None

    Raises:
        ImportError: If BAML client is not available
    """
    if tools is None:
        return None

    if baml_types is None:
        raise ImportError("BAML client not available. Run 'baml update-client'.")

    baml_tools = []
    for tool in tools:
        # Convert function dict to string map for BAML
        function_dict = tool.get('function', {})
        function_map = {str(k): str(v) if not isinstance(v, (dict, list)) else str(v) for k, v in function_dict.items()}
        baml_tools.append(
            baml_types.Tool(
                type=tool.get('type', 'function'),
                function_def=function_map
            )
        )
    return baml_tools


def convert_baml_response_to_model_response(
    baml_response: Any,
) -> ModelResponse:
    """Convert BAML response to LiteLLM ModelResponse format.

    Args:
        baml_response: BAML completion response

    Returns:
        ModelResponse compatible with LiteLLM format

    Raises:
        ImportError: If BAML client is not available
    """
    if baml_types is None:
        raise ImportError("BAML client not available. Run 'baml update-client'.")

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
    content: list[Any]
) -> str | list[dict[str, Any]]:
    """Convert BAML content to OpenHands format.

    Returns either a string (for simple text) or a list of content blocks.

    Args:
        content: List of BAML content objects

    Returns:
        Either a string or a list of content dicts
    """
    if not content:
        return ""

    # Check if we have a single text content (simple case)
    if len(content) == 1:
        item = content[0]
        if hasattr(item, 'type') and item.type == 'text' and hasattr(item, 'text'):
            # Simple text content - return as string
            return item.text

    # Multiple content blocks (vision, etc.)
    result = []
    for item in content:
        if hasattr(item, 'type'):
            if item.type == 'text' and hasattr(item, 'text'):
                content_dict: dict[str, Any] = {
                    'type': 'text',
                    'text': item.text
                }
                if hasattr(item, 'cache_control') and item.cache_control:
                    content_dict['cache_control'] = item.cache_control
                result.append(content_dict)
            elif item.type == 'image_url' and hasattr(item, 'image_url'):
                content_dict = {
                    'type': 'image_url',
                    'image_url': item.image_url
                }
                if hasattr(item, 'cache_control') and item.cache_control:
                    content_dict['cache_control'] = item.cache_control
                result.append(content_dict)

    # If we only have one text block, return as string for compatibility
    if len(result) == 1 and result[0].get('type') == 'text':
        return result[0].get('text', '')

    return result


def _format_messages_to_text(
    messages: list[dict[str, Any]] | list[Message],
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
        if isinstance(msg, Message):
            msg_dict = msg.model_dump()
        else:
            msg_dict = msg

        role = msg_dict.get('role', 'user')
        content = msg_dict.get('content', '')

        # Format content based on type
        if isinstance(content, str):
            content_text = content
        elif isinstance(content, list):
            content_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        content_parts.append(item.get('text', ''))
                    elif item.get('type') == 'image_url':
                        image_url = item.get('image_url', {})
                        if isinstance(image_url, dict):
                            content_parts.append(f"[Image: {image_url.get('url', '')}]")
                else:
                    content_parts.append(str(item))
            content_text = ' '.join(content_parts)
        else:
            content_text = str(content)

        # Format message by role
        if role == 'system':
            conversation_parts.append(f"System: {content_text}")
        elif role == 'user':
            conversation_parts.append(f"User: {content_text}")
        elif role == 'assistant':
            conversation_parts.append(f"Assistant: {content_text}")
        elif role == 'tool':
            name = msg_dict.get('name', 'tool')
            conversation_parts.append(f"Tool ({name}): {content_text}")

        # Add tool calls if present
        if msg_dict.get('tool_calls'):
            tool_calls = msg_dict['tool_calls']
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    func_name = tool_call.get('function', {}).get('name', 'unknown')
                    func_args = tool_call.get('function', {}).get('arguments', '{}')
                    conversation_parts.append(f"Assistant calls function: {func_name}({func_args})")

    # Add tools information if provided
    if tools:
        tools_text = "Available tools:\n"
        for tool in tools:
            if isinstance(tool, dict):
                func_info = tool.get('function', {})
                func_name = func_info.get('name', 'unknown')
                func_desc = func_info.get('description', '')
                func_params = func_info.get('parameters', {})

                # Format function signature with parameters
                tools_text += f"\n{func_name}: {func_desc}\n"

                # Add parameter details if available
                if func_params and isinstance(func_params, dict):
                    properties = func_params.get('properties', {})
                    required = func_params.get('required', [])
                    if properties:
                        tools_text += "  Parameters:\n"
                        for param_name, param_info in properties.items():
                            param_type = param_info.get('type', 'string')
                            param_desc = param_info.get('description', '')
                            is_required = param_name in required
                            req_marker = " (required)" if is_required else " (optional)"
                            tools_text += f"    - {param_name} ({param_type}){req_marker}: {param_desc}\n"
        conversation_parts.append(tools_text)

    return '\n\n'.join(conversation_parts)


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


async def call_baml_completion_async(
    messages: list[dict[str, Any]] | list[Message],
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> ModelResponse:
    """Call BAML completion function asynchronously and return ModelResponse.

    Args:
        messages: List of messages
        tools: Optional list of tools for function calling
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        ModelResponse compatible with LiteLLM format

    Raises:
        ImportError: If BAML client is not available
    """
    if async_b is None:
        raise ImportError("BAML async client not available. Run 'baml update-client'.")

    # Format messages to text for BAML
    conversation_text = _format_messages_to_text(messages, tools)

    # Create BAML request with formatted conversation
    request = baml_types.FormattedLLMRequest(
        conversation=conversation_text,
        temperature=kwargs.get('temperature'),
        max_tokens=kwargs.get('max_tokens') or kwargs.get('max_completion_tokens'),
        top_p=kwargs.get('top_p'),
        top_k=kwargs.get('top_k'),
        seed=kwargs.get('seed'),
        stop=kwargs.get('stop')
    )

    # Call BAML function asynchronously
    baml_response = await async_b.CompleteLLMRequest(request)

    # Convert back to ModelResponse
    return convert_baml_response_to_model_response(baml_response)

