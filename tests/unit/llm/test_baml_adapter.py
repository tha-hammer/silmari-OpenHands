"""Tests for BAML adapter."""

from unittest.mock import MagicMock, patch

import pytest
from litellm.types.utils import ModelResponse

from openhands.core.message import ImageContent, Message, TextContent
from openhands.llm.baml_adapter import (
    _convert_baml_content_to_dict,
    _format_messages_to_text,
    call_baml_completion,
    call_baml_completion_async,
    convert_baml_response_to_model_response,
    convert_messages_to_baml,
    convert_tools_to_baml,
)


@pytest.fixture
def mock_baml_types():
    """Mock BAML types for testing."""
    with patch('openhands.llm.baml_adapter.baml_types') as mock_types:
        # Create mock classes
        mock_text_content = MagicMock()
        mock_text_content.type = 'text'
        mock_text_content.text = 'test'
        mock_text_content.cache_control = None

        mock_image_content = MagicMock()
        mock_image_content.type = 'image_url'
        mock_image_content.image_url = {'url': 'http://example.com/image.jpg'}
        mock_image_content.cache_control = None

        mock_message = MagicMock()
        mock_message.role = 'user'
        mock_message.content = [mock_text_content]
        mock_message.tool_calls = None
        mock_message.tool_call_id = None
        mock_message.name = None

        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.message = mock_message
        mock_choice.finish_reason = 'stop'

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_usage.prompt_tokens_details = None

        mock_response = MagicMock()
        mock_response.id = 'test-id'
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = 'gpt-4o'
        mock_response.created = 1234567890

        # Set up type classes
        mock_types.TextContent = MagicMock(return_value=mock_text_content)
        mock_types.ImageContent = MagicMock(return_value=mock_image_content)
        mock_types.Message = MagicMock(return_value=mock_message)
        mock_types.Tool = MagicMock()
        mock_types.FormattedLLMRequest = MagicMock()

        yield mock_types


@pytest.fixture
def mock_baml_client():
    """Mock BAML sync client."""
    with patch('openhands.llm.baml_adapter.b') as mock_b:
        mock_response = MagicMock()
        mock_response.id = 'test-id'
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].index = 0
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.role = 'assistant'
        mock_response.choices[0].message.content = [MagicMock()]
        mock_response.choices[0].message.content[0].type = 'text'
        mock_response.choices[0].message.content[0].text = 'Test response'
        mock_response.choices[0].finish_reason = 'stop'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.prompt_tokens_details = None
        mock_response.model = 'gpt-4o'
        mock_response.created = 1234567890

        mock_b.CompleteLLMRequest.return_value = mock_response
        yield mock_b


class TestConvertMessagesToBaml:
    """Test message conversion to BAML format."""

    def test_convert_simple_text_message(self, mock_baml_types):
        """Test converting a simple text message."""
        message = Message(role='user', content=[TextContent(text='Hello')])
        result = convert_messages_to_baml([message])

        assert len(result) == 1
        assert result[0].role == 'user'
        mock_baml_types.TextContent.assert_called_once()

    def test_convert_message_dict(self, mock_baml_types):
        """Test converting a message dict."""
        message_dict = {'role': 'user', 'content': 'Hello'}
        result = convert_messages_to_baml([message_dict])

        assert len(result) == 1
        assert result[0].role == 'user'

    def test_convert_message_with_image(self, mock_baml_types):
        """Test converting a message with image content."""
        message = Message(
            role='user',
            content=[
                TextContent(text='Look at this'),
                ImageContent(image_urls=['http://example.com/image.jpg'])
            ]
        )
        result = convert_messages_to_baml([message])

        assert len(result) == 1
        # Should create both TextContent and ImageContent
        assert mock_baml_types.TextContent.called
        assert mock_baml_types.ImageContent.called

    def test_convert_message_with_cache_control(self, mock_baml_types):
        """Test converting a message with cache control."""
        message_dict = {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Hello',
                    'cache_control': {'type': 'ephemeral'}
                }
            ]
        }
        result = convert_messages_to_baml([message_dict])

        assert len(result) == 1
        # Verify cache_control was set
        text_content = mock_baml_types.TextContent.return_value
        assert hasattr(text_content, 'cache_control')

    def test_convert_multiple_messages(self, mock_baml_types):
        """Test converting multiple messages."""
        messages = [
            Message(role='system', content=[TextContent(text='System message')]),
            Message(role='user', content=[TextContent(text='User message')]),
        ]
        result = convert_messages_to_baml(messages)

        assert len(result) == 2
        assert result[0].role == 'system'
        assert result[1].role == 'user'

    def test_convert_message_with_tool_calls(self, mock_baml_types):
        """Test converting a message with tool calls."""
        message = Message(
            role='assistant',
            content=[TextContent(text='Response')],
            tool_calls=[{
                'id': 'call_1',
                'type': 'function',
                'function': {'name': 'test_function', 'arguments': '{}'}
            }]
        )
        result = convert_messages_to_baml([message])

        assert len(result) == 1
        assert result[0].tool_calls is not None

    def test_convert_message_without_baml_client(self):
        """Test that ImportError is raised when BAML client is not available."""
        with patch('openhands.llm.baml_adapter.baml_types', None):
            message = Message(role='user', content=[TextContent(text='Hello')])
            with pytest.raises(ImportError, match='BAML client not available'):
                convert_messages_to_baml([message])


class TestConvertToolsToBaml:
    """Test tool conversion to BAML format."""

    def test_convert_tools(self, mock_baml_types):
        """Test converting tools to BAML format."""
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'test_function',
                    'description': 'A test function',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'param1': {'type': 'string', 'description': 'Param 1'}
                        }
                    }
                }
            }
        ]
        result = convert_tools_to_baml(tools)

        assert result is not None
        assert len(result) == 1
        mock_baml_types.Tool.assert_called_once()

    def test_convert_tools_none(self):
        """Test converting None tools."""
        result = convert_tools_to_baml(None)
        assert result is None

    def test_convert_tools_without_baml_client(self):
        """Test that ImportError is raised when BAML client is not available."""
        with patch('openhands.llm.baml_adapter.baml_types', None):
            tools = [{'type': 'function', 'function': {'name': 'test'}}]
            with pytest.raises(ImportError, match='BAML client not available'):
                convert_tools_to_baml(tools)


class TestConvertBamlResponseToModelResponse:
    """Test BAML response conversion to ModelResponse."""

    def test_convert_simple_response(self, mock_baml_types):
        """Test converting a simple BAML response."""
        # Create a mock response
        mock_response = MagicMock()
        mock_response.id = 'test-id'
        mock_response.model = 'gpt-4o'
        mock_response.created = 1234567890

        # Mock choice
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.finish_reason = 'stop'

        # Mock message
        mock_message = MagicMock()
        mock_message.role = 'assistant'
        mock_message.content = [MagicMock()]
        mock_message.content[0].type = 'text'
        mock_message.content[0].text = 'Test response'
        mock_message.tool_calls = None
        mock_message.tool_call_id = None
        mock_message.name = None

        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # Mock usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_usage.prompt_tokens_details = None
        mock_response.usage = mock_usage

        result = convert_baml_response_to_model_response(mock_response)

        assert isinstance(result, ModelResponse)
        assert result.id == 'test-id'
        assert result.model == 'gpt-4o'
        assert len(result.choices) == 1
        assert result.choices[0]['index'] == 0
        assert result.choices[0]['message']['role'] == 'assistant'
        assert result.choices[0]['message']['content'] == 'Test response'
        assert result.usage is not None
        assert result.usage['prompt_tokens'] == 10
        assert result.usage['completion_tokens'] == 5

    def test_convert_response_with_tool_calls(self, mock_baml_types):
        """Test converting a response with tool calls."""
        mock_response = MagicMock()
        mock_response.id = 'test-id'
        mock_response.model = 'gpt-4o'
        mock_response.created = 1234567890

        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.finish_reason = 'stop'

        mock_message = MagicMock()
        mock_message.role = 'assistant'
        mock_message.content = []
        mock_message.tool_calls = [{'id': 'call_1', 'type': 'function'}]
        mock_message.tool_call_id = None
        mock_message.name = None

        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        result = convert_baml_response_to_model_response(mock_response)

        assert result.choices[0]['message']['tool_calls'] == [{'id': 'call_1', 'type': 'function'}]

    def test_convert_response_without_baml_client(self):
        """Test that ImportError is raised when BAML client is not available."""
        with patch('openhands.llm.baml_adapter.baml_types', None):
            mock_response = MagicMock()
            with pytest.raises(ImportError, match='BAML client not available'):
                convert_baml_response_to_model_response(mock_response)


class TestConvertBamlContentToDict:
    """Test BAML content conversion to dict."""

    def test_convert_single_text_content(self):
        """Test converting single text content to string."""
        mock_content = MagicMock()
        mock_content.type = 'text'
        mock_content.text = 'Hello'

        result = _convert_baml_content_to_dict([mock_content])

        assert result == 'Hello'

    def test_convert_multiple_content_blocks(self):
        """Test converting multiple content blocks."""
        mock_text = MagicMock()
        mock_text.type = 'text'
        mock_text.text = 'Hello'

        mock_image = MagicMock()
        mock_image.type = 'image_url'
        mock_image.image_url = {'url': 'http://example.com/image.jpg'}

        result = _convert_baml_content_to_dict([mock_text, mock_image])

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['type'] == 'text'
        assert result[0]['text'] == 'Hello'
        assert result[1]['type'] == 'image_url'

    def test_convert_empty_content(self):
        """Test converting empty content."""
        result = _convert_baml_content_to_dict([])
        assert result == ''

    def test_convert_content_with_cache_control(self):
        """Test converting content with cache control."""
        mock_content = MagicMock()
        mock_content.type = 'text'
        mock_content.text = 'Hello'
        mock_content.cache_control = {'type': 'ephemeral'}

        result = _convert_baml_content_to_dict([mock_content])

        # Single text should return as string
        assert result == 'Hello'


class TestFormatMessagesToText:
    """Test message formatting to text."""

    def test_format_simple_messages(self):
        """Test formatting simple text messages."""
        messages = [
            Message(role='user', content=[TextContent(text='Hello')]),
            Message(role='assistant', content=[TextContent(text='Hi there')]),
        ]
        result = _format_messages_to_text(messages)

        assert 'User: Hello' in result
        assert 'Assistant: Hi there' in result

    def test_format_messages_with_tools(self):
        """Test formatting messages with tools."""
        messages = [
            Message(role='user', content=[TextContent(text='Execute ls')]),
        ]
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'execute_bash',
                    'description': 'Execute a bash command',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'command': {
                                'type': 'string',
                                'description': 'The command to execute'
                            }
                        },
                        'required': ['command']
                    }
                }
            }
        ]
        result = _format_messages_to_text(messages, tools)

        assert 'User: Execute ls' in result
        assert 'Available tools:' in result
        assert 'execute_bash' in result
        assert 'Parameters:' in result
        assert 'command (string) (required)' in result

    def test_format_messages_with_image(self):
        """Test formatting messages with image content."""
        messages = [
            Message(
                role='user',
                content=[
                    TextContent(text='Look at this'),
                    ImageContent(image_urls=['http://example.com/image.jpg'])
                ]
            ),
        ]
        result = _format_messages_to_text(messages)

        assert 'User: Look at this' in result
        assert '[Image:' in result

    def test_format_messages_with_tool_calls(self):
        """Test formatting messages with tool calls."""
        messages = [
            Message(
                role='assistant',
                content=[TextContent(text='I will call a function')],
                tool_calls=[{
                    'id': 'call_1',
                    'type': 'function',
                    'function': {
                        'name': 'test_function',
                        'arguments': '{"param": "value"}'
                    }
                }]
            ),
        ]
        result = _format_messages_to_text(messages)

        assert 'Assistant: I will call a function' in result
        assert 'Assistant calls function: test_function' in result

    def test_format_message_dicts(self):
        """Test formatting message dicts."""
        messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi'},
        ]
        result = _format_messages_to_text(messages)

        assert 'User: Hello' in result
        assert 'Assistant: Hi' in result


class TestCallBamlCompletion:
    """Test BAML completion calls."""

    @patch('openhands.llm.baml_adapter._format_messages_to_text')
    @patch('openhands.llm.baml_adapter.convert_baml_response_to_model_response')
    def test_call_baml_completion_success(
        self,
        mock_convert_response,
        mock_format_text,
        mock_baml_client,
        mock_baml_types
    ):
        """Test successful BAML completion call."""
        # Setup mocks
        mock_format_text.return_value = 'Formatted conversation'
        mock_response = ModelResponse(
            id='test-id',
            choices=[{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'Test response'},
                'finish_reason': 'stop'
            }],
            usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
            model='gpt-4o'
        )
        mock_convert_response.return_value = mock_response

        # Create test messages
        messages = [Message(role='user', content=[TextContent(text='Hello')])]

        # Call function
        result = call_baml_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )

        # Verify calls
        mock_format_text.assert_called_once_with(messages, None)
        mock_baml_types.FormattedLLMRequest.assert_called_once()
        mock_baml_client.CompleteLLMRequest.assert_called_once()
        mock_convert_response.assert_called_once()

        assert isinstance(result, ModelResponse)

    def test_call_baml_completion_without_client(self):
        """Test that ImportError is raised when BAML client is not available."""
        with patch('openhands.llm.baml_adapter.b', None):
            messages = [Message(role='user', content=[TextContent(text='Hello')])]
            with pytest.raises(ImportError, match='BAML client not available'):
                call_baml_completion(messages=messages)

    @patch('openhands.llm.baml_adapter._format_messages_to_text')
    def test_call_baml_completion_with_tools(
        self,
        mock_format_text,
        mock_baml_client,
        mock_baml_types
    ):
        """Test BAML completion call with tools."""
        mock_format_text.return_value = 'Formatted conversation with tools'

        messages = [Message(role='user', content=[TextContent(text='Execute ls')])]
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'execute_bash',
                    'description': 'Execute a command'
                }
            }
        ]

        call_baml_completion(messages=messages, tools=tools)

        # Verify tools were included in formatting
        mock_format_text.assert_called_once_with(messages, tools)

    @patch('openhands.llm.baml_adapter._format_messages_to_text')
    @patch('openhands.llm.baml_adapter.convert_baml_response_to_model_response')
    def test_call_baml_completion_with_kwargs(
        self,
        mock_convert_response,
        mock_format_text,
        mock_baml_client,
        mock_baml_types
    ):
        """Test BAML completion call with various kwargs."""
        mock_format_text.return_value = 'Formatted conversation'
        mock_convert_response.return_value = ModelResponse(
            id='test-id',
            choices=[],
            model='gpt-4o'
        )

        messages = [Message(role='user', content=[TextContent(text='Hello')])]

        call_baml_completion(
            messages=messages,
            temperature=0.8,
            max_tokens=200,
            top_p=0.9,
            top_k=40,
            seed=42,
            stop=['\n', 'STOP']
        )

        # Verify FormattedLLMRequest was called with all kwargs
        call_args = mock_baml_types.FormattedLLMRequest.call_args
        assert call_args[1]['temperature'] == 0.8
        assert call_args[1]['max_tokens'] == 200
        assert call_args[1]['top_p'] == 0.9
        assert call_args[1]['top_k'] == 40
        assert call_args[1]['seed'] == 42
        assert call_args[1]['stop'] == ['\n', 'STOP']


@pytest.fixture
def mock_baml_async_client():
    """Mock BAML async client."""
    with patch('openhands.llm.baml_adapter.async_b') as mock_async_b:
        mock_response = MagicMock()
        mock_response.id = 'test-id'
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].index = 0
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.role = 'assistant'
        mock_response.choices[0].message.content = [MagicMock()]
        mock_response.choices[0].message.content[0].type = 'text'
        mock_response.choices[0].message.content[0].text = 'Test async response'
        mock_response.choices[0].finish_reason = 'stop'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.prompt_tokens_details = None
        mock_response.model = 'gpt-4o'
        mock_response.created = 1234567890

        # Make it async
        async def async_complete(*args, **kwargs):
            return mock_response

        mock_async_b.CompleteLLMRequest = async_complete
        yield mock_async_b


class TestCallBamlCompletionAsync:
    """Test async BAML completion calls."""

    @pytest.mark.asyncio
    @patch('openhands.llm.baml_adapter._format_messages_to_text')
    @patch('openhands.llm.baml_adapter.convert_baml_response_to_model_response')
    async def test_call_baml_completion_async_success(
        self,
        mock_convert_response,
        mock_format_text,
        mock_baml_async_client,
        mock_baml_types
    ):
        """Test successful async BAML completion call."""
        # Setup mocks
        mock_format_text.return_value = 'Formatted conversation'
        mock_response = ModelResponse(
            id='test-id',
            choices=[{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'Test async response'},
                'finish_reason': 'stop'
            }],
            usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
            model='gpt-4o'
        )
        mock_convert_response.return_value = mock_response

        # Create test messages
        messages = [Message(role='user', content=[TextContent(text='Hello')])]

        # Call async function
        result = await call_baml_completion_async(
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )

        # Verify calls
        mock_format_text.assert_called_once_with(messages, None)
        mock_baml_types.FormattedLLMRequest.assert_called_once()
        mock_convert_response.assert_called_once()

        assert isinstance(result, ModelResponse)
        assert result.id == 'test-id'

    @pytest.mark.asyncio
    async def test_call_baml_completion_async_without_client(self):
        """Test that ImportError is raised when BAML async client is not available."""
        with patch('openhands.llm.baml_adapter.async_b', None):
            messages = [Message(role='user', content=[TextContent(text='Hello')])]
            with pytest.raises(ImportError, match='BAML async client not available'):
                await call_baml_completion_async(messages=messages)

    @pytest.mark.asyncio
    @patch('openhands.llm.baml_adapter._format_messages_to_text')
    async def test_call_baml_completion_async_with_tools(
        self,
        mock_format_text,
        mock_baml_async_client,
        mock_baml_types
    ):
        """Test async BAML completion call with tools."""
        mock_format_text.return_value = 'Formatted conversation with tools'

        messages = [Message(role='user', content=[TextContent(text='Execute ls')])]
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'execute_bash',
                    'description': 'Execute a command'
                }
            }
        ]

        await call_baml_completion_async(messages=messages, tools=tools)

        # Verify tools were included in formatting
        mock_format_text.assert_called_once_with(messages, tools)

    @pytest.mark.asyncio
    @patch('openhands.llm.baml_adapter._format_messages_to_text')
    @patch('openhands.llm.baml_adapter.convert_baml_response_to_model_response')
    async def test_call_baml_completion_async_with_kwargs(
        self,
        mock_convert_response,
        mock_format_text,
        mock_baml_async_client,
        mock_baml_types
    ):
        """Test async BAML completion call with various kwargs."""
        mock_format_text.return_value = 'Formatted conversation'
        mock_convert_response.return_value = ModelResponse(
            id='test-id',
            choices=[],
            model='gpt-4o'
        )

        messages = [Message(role='user', content=[TextContent(text='Hello')])]

        await call_baml_completion_async(
            messages=messages,
            temperature=0.8,
            max_tokens=200,
            top_p=0.9,
            top_k=40,
            seed=42,
            stop=['\n', 'STOP']
        )

        # Verify FormattedLLMRequest was called with all kwargs
        call_args = mock_baml_types.FormattedLLMRequest.call_args
        assert call_args[1]['temperature'] == 0.8
        assert call_args[1]['max_tokens'] == 200
        assert call_args[1]['top_p'] == 0.9
        assert call_args[1]['top_k'] == 40
        assert call_args[1]['seed'] == 42
        assert call_args[1]['stop'] == ['\n', 'STOP']

