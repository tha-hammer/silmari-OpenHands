"""Tests for BAML adapter."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from agent_harness.llm.baml_adapter import (
    _format_messages_to_text,
    convert_baml_response_to_model_response,
    call_baml_completion,
    call_baml_completion_async,
)


def test_format_messages_to_text_simple():
    """Test formatting simple messages."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    
    result = _format_messages_to_text(messages)
    
    assert "Hello" in result
    assert "Hi there" in result
    assert "user" in result.lower() or "User" in result
    assert "assistant" in result.lower() or "Assistant" in result


def test_format_messages_to_text_with_list_content():
    """Test formatting messages with list content."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "image_url": {"url": "data:image/png;base64,..."}},
            ],
        }
    ]
    
    result = _format_messages_to_text(messages)
    
    assert "Hello" in result
    assert "image" in result.lower()


def test_format_messages_to_text_with_tools():
    """Test formatting messages with tools."""
    messages = [{"role": "user", "content": "Use the tool"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "First parameter",
                        },
                        "param2": {"type": "number"},
                    },
                    "required": ["param1"],
                },
            },
        }
    ]
    
    result = _format_messages_to_text(messages, tools)
    
    assert "test_tool" in result
    assert "A test tool" in result
    assert "param1" in result
    assert "First parameter" in result
    assert "required" in result.lower() or "(required)" in result


def test_format_messages_to_text_with_tool_calls():
    """Test formatting messages with tool calls."""
    messages = [
        {"role": "user", "content": "Use the tool"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": '{"param": "value"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": "Tool result",
            "tool_call_id": "call_1",
        },
    ]
    
    result = _format_messages_to_text(messages)
    
    assert "test_tool" in result
    assert "Tool result" in result


@patch("agent_harness.llm.baml_adapter.b")
@patch("agent_harness.llm.baml_adapter.baml_types")
def test_convert_baml_response_to_model_response_simple(mock_types, mock_b):
    """Test converting simple BAML response to ModelResponse."""
    # Create a mock BAML response
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_response.tool_calls = None
    mock_response.prompt_tokens = 10
    mock_response.completion_tokens = 5
    mock_response.total_tokens = 15
    
    result = convert_baml_response_to_model_response(mock_response)
    
    assert result["choices"][0]["message"]["content"] == "Test response"
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5
    assert result["usage"]["total_tokens"] == 15
    assert result["choices"][0]["finish_reason"] == "stop"


@patch("agent_harness.llm.baml_adapter.b")
@patch("agent_harness.llm.baml_adapter.baml_types")
def test_convert_baml_response_with_tool_calls(mock_types, mock_b):
    """Test converting BAML response with tool calls."""
    # Create a mock BAML response with tool calls
    mock_response = Mock()
    mock_response.content = None
    mock_response.tool_calls = {
        "call_1": json.dumps({"name": "test_tool", "arguments": {"param": "value"}})
    }
    mock_response.prompt_tokens = None
    mock_response.completion_tokens = None
    mock_response.total_tokens = None
    
    result = convert_baml_response_to_model_response(mock_response)
    
    assert "tool_calls" in result["choices"][0]["message"]
    tool_calls = result["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_1"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "test_tool"


@patch("agent_harness.llm.baml_adapter.b")
@patch("agent_harness.llm.baml_adapter.baml_types")
def test_convert_baml_response_no_usage(mock_types, mock_b):
    """Test converting BAML response without usage info."""
    mock_response = Mock()
    mock_response.content = "Test"
    mock_response.tool_calls = None
    mock_response.prompt_tokens = None
    mock_response.completion_tokens = None
    mock_response.total_tokens = None
    
    result = convert_baml_response_to_model_response(mock_response)
    
    assert result["usage"] is None


@patch("agent_harness.llm.baml_adapter.b")
@patch("agent_harness.llm.baml_adapter.baml_types")
def test_call_baml_completion(mock_types, mock_b):
    """Test calling BAML completion function."""
    # Mock BAML client and types
    mock_request = Mock()
    mock_types.FormattedLLMRequest = Mock(return_value=mock_request)
    
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_response.tool_calls = None
    mock_response.prompt_tokens = 10
    mock_response.completion_tokens = 5
    mock_response.total_tokens = 15
    
    mock_b.CompleteLLMRequest.return_value = mock_response
    
    messages = [{"role": "user", "content": "test"}]
    result = call_baml_completion(messages, temperature=0.7, max_tokens=100)
    
    # Verify BAML function was called
    mock_b.CompleteLLMRequest.assert_called_once()
    assert result["choices"][0]["message"]["content"] == "Test response"


@patch("agent_harness.llm.baml_adapter.b")
def test_call_baml_completion_no_client(mock_b):
    """Test calling BAML completion when client is not available."""
    # Set b to None to simulate missing client
    import agent_harness.llm.baml_adapter as adapter
    original_b = adapter.b
    adapter.b = None
    
    try:
        with pytest.raises(ImportError, match="BAML client not available"):
            call_baml_completion([{"role": "user", "content": "test"}])
    finally:
        adapter.b = original_b


@patch("agent_harness.llm.baml_adapter.async_b")
@patch("agent_harness.llm.baml_adapter.baml_types")
@pytest.mark.asyncio
async def test_call_baml_completion_async(mock_types, mock_async_b):
    """Test calling BAML completion function asynchronously."""
    # Mock BAML async client and types
    mock_request = Mock()
    mock_types.FormattedLLMRequest = Mock(return_value=mock_request)
    
    mock_response = Mock()
    mock_response.content = "Async test response"
    mock_response.tool_calls = None
    mock_response.prompt_tokens = 10
    mock_response.completion_tokens = 5
    mock_response.total_tokens = 15
    
    mock_async_b.CompleteLLMRequest = MagicMock(return_value=mock_response)
    
    messages = [{"role": "user", "content": "test"}]
    result = await call_baml_completion_async(messages, temperature=0.7, max_tokens=100)
    
    # Verify async BAML function was called
    mock_async_b.CompleteLLMRequest.assert_called_once()
    assert result["choices"][0]["message"]["content"] == "Async test response"


@patch("agent_harness.llm.baml_adapter.async_b")
@pytest.mark.asyncio
async def test_call_baml_completion_async_no_client(mock_async_b):
    """Test calling async BAML completion when client is not available."""
    # Set async_b to None to simulate missing client
    import agent_harness.llm.baml_adapter as adapter
    original_async_b = adapter.async_b
    adapter.async_b = None
    
    try:
        with pytest.raises(ImportError, match="BAML async client not available"):
            await call_baml_completion_async([{"role": "user", "content": "test"}])
    finally:
        adapter.async_b = original_async_b

