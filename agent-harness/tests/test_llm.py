"""Tests for LLM integration."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from agent_harness.config import LLMConfig
from agent_harness.llm.llm import LLM


@pytest.fixture
def llm_config():
    """Create a test LLM config."""
    return LLMConfig(
        model="test-model",
        api_key="test-key",
        base_url="https://api.test.com",
        temperature=0.7,
        max_tokens=1000,
    )


def test_llm_initialization(llm_config):
    """Test LLM can be initialized."""
    llm = LLM(llm_config)
    
    assert llm.config == llm_config
    assert llm.use_baml is True
    assert llm.service_id == "agent"


def test_llm_initialization_custom_service_id(llm_config):
    """Test LLM can be initialized with custom service ID."""
    llm = LLM(llm_config, service_id="custom-service")
    
    assert llm.service_id == "custom-service"


def test_llm_sets_environment_variables(llm_config):
    """Test LLM sets environment variables for BAML."""
    # Clear any existing env vars
    for key in ["BAML_API_KEY", "BAML_BASE_URL", "BAML_MODEL"]:
        os.environ.pop(key, None)
    
    llm = LLM(llm_config)
    
    assert os.environ.get("BAML_API_KEY") == "test-key"
    assert os.environ.get("BAML_BASE_URL") == "https://api.test.com"
    assert os.environ.get("BAML_MODEL") == "test-model"


def test_llm_sets_empty_api_key_when_none(llm_config):
    """Test LLM sets empty string for API key when None."""
    config = LLMConfig(model="test-model", api_key=None)
    
    # Clear any existing env vars
    os.environ.pop("BAML_API_KEY", None)
    
    llm = LLM(config)
    
    assert os.environ.get("BAML_API_KEY") == ""


def test_llm_removes_base_url_when_none(llm_config):
    """Test LLM removes BAML_BASE_URL when base_url is None."""
    config = LLMConfig(model="test-model", base_url=None)
    
    # Set it first
    os.environ["BAML_BASE_URL"] = "old-url"
    
    llm = LLM(config)
    
    assert "BAML_BASE_URL" not in os.environ or os.environ.get("BAML_BASE_URL") is None


@patch("agent_harness.llm.llm.call_baml_completion")
def test_llm_completion_sync(mock_baml_completion, llm_config):
    """Test synchronous completion call."""
    mock_response = {
        "choices": [{"message": {"role": "assistant", "content": "test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock_baml_completion.return_value = mock_response
    
    llm = LLM(llm_config)
    messages = [{"role": "user", "content": "test"}]
    
    result = llm.completion(messages)
    
    assert result == mock_response
    mock_baml_completion.assert_called_once()
    call_kwargs = mock_baml_completion.call_args[1]
    assert call_kwargs["messages"] == messages
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 1000


@patch("agent_harness.llm.llm.call_baml_completion")
def test_llm_completion_with_tools(mock_baml_completion, llm_config):
    """Test completion call with tools."""
    mock_response = {
        "choices": [{"message": {"role": "assistant", "content": "test"}}],
    }
    mock_baml_completion.return_value = mock_response
    
    llm = LLM(llm_config)
    messages = [{"role": "user", "content": "test"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    
    result = llm.completion(messages, tools=tools)
    
    assert result == mock_response
    call_kwargs = mock_baml_completion.call_args[1]
    assert call_kwargs["tools"] == tools


@patch("agent_harness.llm.llm.call_baml_completion")
def test_llm_completion_overrides_temperature(mock_baml_completion, llm_config):
    """Test completion call can override temperature."""
    mock_response = {"choices": [{"message": {"content": "test"}}]}
    mock_baml_completion.return_value = mock_response
    
    llm = LLM(llm_config)
    messages = [{"role": "user", "content": "test"}]
    
    result = llm.completion(messages, temperature=0.9)
    
    call_kwargs = mock_baml_completion.call_args[1]
    assert call_kwargs["temperature"] == 0.9


@patch("agent_harness.llm.llm.call_baml_completion_async")
@pytest.mark.asyncio
async def test_llm_completion_async(mock_baml_completion_async, llm_config):
    """Test asynchronous completion call."""
    mock_response = {
        "choices": [{"message": {"role": "assistant", "content": "test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock_baml_completion_async.return_value = mock_response
    
    llm = LLM(llm_config)
    messages = [{"role": "user", "content": "test"}]
    
    result = await llm.async_completion(messages)
    
    assert result == mock_response
    mock_baml_completion_async.assert_called_once()
    call_kwargs = mock_baml_completion_async.call_args[1]
    assert call_kwargs["messages"] == messages
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 1000


@patch("agent_harness.llm.llm.call_baml_completion_async")
@pytest.mark.asyncio
async def test_llm_completion_async_with_tools(mock_baml_completion_async, llm_config):
    """Test async completion call with tools."""
    mock_response = {
        "choices": [{"message": {"role": "assistant", "content": "test"}}],
    }
    mock_baml_completion_async.return_value = mock_response
    
    llm = LLM(llm_config)
    messages = [{"role": "user", "content": "test"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    
    result = await llm.async_completion(messages, tools=tools)
    
    assert result == mock_response
    call_kwargs = mock_baml_completion_async.call_args[1]
    assert call_kwargs["tools"] == tools

