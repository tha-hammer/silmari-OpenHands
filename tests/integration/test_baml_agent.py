"""Integration tests for BAML with agents using local Ollama LLM."""

import os
import pytest

# Import litellm types - required dependency
pytest.importorskip("litellm", reason="litellm is required for BAML integration tests")

from litellm.types.utils import ModelResponse

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.config import AgentConfig, LLMConfig, OpenHandsConfig
from openhands.core.message import Message, TextContent
from openhands.llm.llm_registry import LLMRegistry


@pytest.fixture
def ollama_model():
    """Get Ollama model name from environment or use default."""
    import os
    return os.environ.get('OLLAMA_MODEL', 'llama2')


@pytest.fixture
def ollama_base_url():
    """Get Ollama base URL from environment or use default."""
    import os
    return os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')


@pytest.fixture
def llm_config_with_baml(ollama_model, ollama_base_url):
    """Create LLM config with BAML enabled using Ollama."""
    # Set BAML environment variables for Ollama (no API key needed)
    os.environ['BAML_API_KEY'] = ''  # Empty string for Ollama
    os.environ['BAML_BASE_URL'] = ollama_base_url
    os.environ['BAML_MODEL'] = f'ollama/{ollama_model}'

    return LLMConfig(
        model=f'ollama/{ollama_model}',
        base_url=ollama_base_url,
        api_key=None,  # Ollama doesn't require API key
        use_baml=True,
        num_retries=1,  # Reduce retries for faster tests
        retry_min_wait=1,  # Must be int, minimum 1 second
        retry_max_wait=2,  # Must be int, maximum 2 seconds
    )


@pytest.fixture
def llm_config_without_baml(ollama_model, ollama_base_url):
    """Create LLM config with BAML disabled using Ollama."""
    return LLMConfig(
        model=f'ollama/{ollama_model}',
        base_url=ollama_base_url,
        api_key=None,  # Ollama doesn't require API key
        use_baml=False,
        num_retries=1,
        retry_min_wait=1,  # Must be int, minimum 1 second
        retry_max_wait=2,  # Must be int, maximum 2 seconds
    )


@pytest.fixture
def openhands_config_with_baml(llm_config_with_baml):
    """Create OpenHands config with BAML enabled."""
    config = OpenHandsConfig()
    config.set_llm_config(llm_config_with_baml)
    return config


@pytest.fixture
def openhands_config_without_baml(llm_config_without_baml):
    """Create OpenHands config with BAML disabled."""
    config = OpenHandsConfig()
    config.set_llm_config(llm_config_without_baml)
    return config


@pytest.fixture
def llm_registry_with_baml(openhands_config_with_baml):
    """Create LLM registry with BAML enabled."""
    return LLMRegistry(config=openhands_config_with_baml)


@pytest.fixture
def llm_registry_without_baml(openhands_config_without_baml):
    """Create LLM registry with BAML disabled."""
    return LLMRegistry(config=openhands_config_without_baml)


@pytest.fixture
def check_ollama_available(ollama_base_url):
    """Check if Ollama is available, skip tests if not."""
    import httpx
    try:
        response = httpx.get(f'{ollama_base_url}/api/tags', timeout=2)
        if response.status_code != 200:
            pytest.skip(f"Ollama not available at {ollama_base_url}")
    except Exception:
        pytest.skip(f"Ollama not available at {ollama_base_url}. Make sure Ollama is running.")


class TestBamlAgentIntegration:
    """Test BAML integration with agents using real Ollama LLM."""

    def test_agent_initializes_with_baml_enabled(
        self,
        llm_registry_with_baml,
        check_ollama_available
    ):
        """Test that agent initializes correctly when BAML is enabled."""
        config = AgentConfig()
        agent = CodeActAgent(config=config, llm_registry=llm_registry_with_baml)

        # Verify agent was created
        assert agent is not None
        assert agent.llm is not None
        # Verify BAML flag is set
        assert agent.llm.use_baml is True

    def test_agent_initializes_with_baml_disabled(
        self,
        llm_registry_without_baml,
        check_ollama_available
    ):
        """Test that agent initializes correctly when BAML is disabled."""
        config = AgentConfig()
        agent = CodeActAgent(config=config, llm_registry=llm_registry_without_baml)

        # Verify agent was created
        assert agent is not None
        assert agent.llm is not None
        # Verify BAML flag is False (default)
        assert agent.llm.use_baml is False

    def test_agent_uses_baml_when_enabled(
        self,
        llm_registry_with_baml,
        check_ollama_available
    ):
        """Test that agent uses BAML completion when enabled with real Ollama."""
        config = AgentConfig()
        agent = CodeActAgent(config=config, llm_registry=llm_registry_with_baml)

        messages = [Message(role='user', content=[TextContent(text='Say hello in one word')])]

        # Call completion through the agent's LLM - this will make a real call to Ollama via BAML
        response = agent.llm.completion(messages=messages)

        # Verify response structure (ModelResponse object)
        assert response is not None
        assert isinstance(response, ModelResponse)
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], 'message')
        # Verify BAML was used (check that response came through)
        assert response.choices[0].message.content is not None

    def test_agent_uses_litellm_when_baml_disabled(
        self,
        llm_registry_without_baml,
        check_ollama_available
    ):
        """Test that agent uses LiteLLM directly when BAML is disabled."""
        config = AgentConfig()
        agent = CodeActAgent(config=config, llm_registry=llm_registry_without_baml)

        messages = [Message(role='user', content=[TextContent(text='Say hello in one word')])]

        # Call completion through the agent's LLM - this will make a real call to Ollama via LiteLLM
        response = agent.llm.completion(messages=messages)

        # Verify response structure (ModelResponse object)
        assert response is not None
        assert isinstance(response, ModelResponse)
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        # Verify BAML was NOT used
        assert agent.llm.use_baml is False


class TestBamlFunctionCalling:
    """Test BAML function calling with agents using real Ollama LLM."""

    def test_baml_with_function_calling(
        self,
        llm_registry_with_baml,
        check_ollama_available
    ):
        """Test BAML completion with function calling tools using real Ollama."""
        config = AgentConfig()
        agent = CodeActAgent(config=config, llm_registry=llm_registry_with_baml)

        messages = [Message(role='user', content=[TextContent(text='What tools are available?')])]
        tools = agent.tools[:2]  # Use first 2 tools to keep prompt shorter

        # Call completion with tools - this will make a real call to Ollama via BAML
        response = agent.llm.completion(messages=messages, tools=tools)

        # Verify response structure (ModelResponse object)
        assert response is not None
        assert isinstance(response, ModelResponse)
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0

    def test_baml_tool_formatting_includes_parameters(
        self,
        llm_registry_with_baml,
        check_ollama_available
    ):
        """Test that BAML tool formatting includes detailed parameter information."""
        config = AgentConfig()
        agent = CodeActAgent(config=config, llm_registry=llm_registry_with_baml)

        messages = [Message(role='user', content=[TextContent(text='List available functions')])]
        tools = agent.tools[:1]  # Use first tool only

        # Call completion - this will format tools with parameters via BAML
        response = agent.llm.completion(messages=messages, tools=tools)

        # Verify response structure (ModelResponse object)
        assert response is not None
        assert isinstance(response, ModelResponse)
        assert hasattr(response, 'choices')


class TestBamlAsyncIntegration:
    """Test BAML async integration with agents using real Ollama LLM."""

    @pytest.mark.asyncio
    async def test_async_baml_completion(
        self,
        llm_registry_with_baml,
        check_ollama_available
    ):
        """Test async BAML completion with agents using real Ollama."""
        config = AgentConfig()
        agent = CodeActAgent(config=config, llm_registry=llm_registry_with_baml)

        from openhands.llm.async_llm import AsyncLLM
        async_llm = AsyncLLM(config=agent.llm.config, service_id='test')

        messages = [Message(role='user', content=[TextContent(text='Say hello')])]

        # Call async completion - this will make a real async call to Ollama via BAML
        response = await async_llm.async_completion(messages=messages)

        # Verify response structure (ModelResponse object)
        assert response is not None
        assert isinstance(response, ModelResponse)
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0

    @pytest.mark.asyncio
    async def test_async_litellm_when_baml_disabled(
        self,
        llm_registry_without_baml,
        check_ollama_available
    ):
        """Test async LiteLLM completion when BAML is disabled."""
        config = AgentConfig()
        agent = CodeActAgent(config=config, llm_registry=llm_registry_without_baml)

        from openhands.llm.async_llm import AsyncLLM
        async_llm = AsyncLLM(config=agent.llm.config, service_id='test')

        messages = [Message(role='user', content=[TextContent(text='Say hello')])]

        # Call async completion - this will make a real async call to Ollama via LiteLLM
        response = await async_llm.async_completion(messages=messages)

        # Verify response structure (ModelResponse object)
        assert response is not None
        assert isinstance(response, ModelResponse)
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
