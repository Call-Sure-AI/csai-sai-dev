import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List, Dict
import asyncio

# The path to the module we are testing
CLAUDE_SERVICE_PATH = "src.api.claude.service"

# We need to import the functions to be tested
from src.api.claude.service import (
    stream_text_response,
    generate_full_response,
    _prepare_claude_messages,
    DEFAULT_CLAUDE_MODEL
)

# --- Test Data Fixtures ---

@pytest.fixture
def valid_user_message() -> List[Dict[str, str]]:
    """A valid message list starting with a user message."""
    return [{"role": "user", "content": "Hello, Claude!"}]


@pytest.fixture
def valid_system_message() -> List[Dict[str, str]]:
    """A valid message list starting with a system prompt."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, Claude!"}
    ]

@pytest.fixture
def invalid_message_no_user() -> List[Dict[str, str]]:
    """An invalid message list without a user message."""
    return [{"role": "assistant", "content": "How can I help?"}]

@pytest.fixture
def invalid_message_system_only() -> List[Dict[str, str]]:
    """An invalid message list with only a system prompt."""
    return [{"role": "system", "content": "You are helpful."}]


# --- Tests for _prepare_claude_messages Helper Function ---

def test_prepare_messages_with_system_prompt(valid_system_message):
    """Should correctly extract system prompt and conversation messages."""
    system_prompt, messages = _prepare_claude_messages(valid_system_message)
    assert system_prompt == "You are a helpful assistant."
    assert messages == [{"role": "user", "content": "Hello, Claude!"}]

def test_prepare_messages_without_system_prompt(valid_user_message):
    """Should handle lists without a system prompt correctly."""
    system_prompt, messages = _prepare_claude_messages(valid_user_message)
    assert system_prompt is None
    assert messages == valid_user_message

def test_prepare_messages_invalid_start_role(invalid_message_no_user):
    """Should raise ValueError if the conversation doesn't start with a user role."""
    with pytest.raises(ValueError, match="must be from the 'user'"):
        _prepare_claude_messages(invalid_message_no_user)

def test_prepare_messages_empty_list():
    """Should raise ValueError for an empty message list."""
    with pytest.raises(ValueError, match="cannot be empty"):
        _prepare_claude_messages([])

def test_prepare_messages_system_only(invalid_message_system_only):
    """Should raise ValueError if only a system prompt is provided."""
    with pytest.raises(ValueError, match="must be from the 'user'"):
        _prepare_claude_messages(invalid_message_system_only)


# --- Tests for stream_text_response ---

@pytest.mark.asyncio
async def test_stream_text_response_success(valid_user_message):
    """Should stream text chunks successfully for a valid request."""
    # Mock the async context manager for the stream
    mock_stream = MagicMock()

    # Define the mock events the stream will yield
    mock_events = [
        MagicMock(type="message_start"),
        MagicMock(type="content_block_delta", delta=MagicMock(type="text_delta", text="Hello")),
        MagicMock(type="content_block_delta", delta=MagicMock(type="text_delta", text=", ")),
        MagicMock(type="content_block_delta", delta=MagicMock(type="text_delta", text="world!")),
        MagicMock(type="message_stop")
    ]
    
    # Create an async iterator from the mock events
    async def event_generator():
        for event in mock_events:
            yield event

    mock_stream.__aenter__.return_value = event_generator()

    # Patch the async_client in the service module
    with patch(f"{CLAUDE_SERVICE_PATH}.async_client.messages.stream", return_value=mock_stream) as mock_api_call:
        # Collect the results from the async generator
        result_chunks = [chunk async for chunk in stream_text_response(valid_user_message)]
        
        # Assertions
        mock_api_call.assert_called_once_with(
            model=DEFAULT_CLAUDE_MODEL,
            messages=valid_user_message,
            system=None,
            temperature=0.7,
            max_tokens=4096
        )
        assert "".join(result_chunks) == "Hello, world!"
        assert len(result_chunks) == 3


@pytest.mark.asyncio
async def test_stream_text_with_system_prompt(valid_system_message):
    """Should correctly handle a stream request with a system prompt."""
    mock_stream = MagicMock()
    mock_events = [MagicMock(type="content_block_delta", delta=MagicMock(type="text_delta", text="Yes."))]
    
    async def event_generator():
        for event in mock_events:
            yield event

    mock_stream.__aenter__.return_value = event_generator()
    
    with patch(f"{CLAUDE_SERVICE_PATH}.async_client.messages.stream", return_value=mock_stream) as mock_api_call:
        results = [chunk async for chunk in stream_text_response(valid_system_message)]

        # Check that the system prompt was correctly separated
        mock_api_call.assert_called_once()
        call_args, call_kwargs = mock_api_call.call_args
        assert call_kwargs['system'] == "You are a helpful assistant."
        assert call_kwargs['messages'] == [{"role": "user", "content": "Hello, Claude!"}]
        assert "".join(results) == "Yes."


# --- Tests for generate_full_response ---

@pytest.mark.asyncio
async def test_generate_full_response_success(valid_user_message):
    """Should return a complete text response successfully."""
    # Create a mock response object that mimics the Anthropic API response
    mock_api_response = MagicMock()
    mock_api_response.content = [MagicMock(text=" This is a full response. ")]
    
    with patch(f"{CLAUDE_SERVICE_PATH}.async_client.messages.create", new=AsyncMock(return_value=mock_api_response)) as mock_api_call:
        result = await generate_full_response(valid_user_message)
        
        mock_api_call.assert_called_once_with(
            model=DEFAULT_CLAUDE_MODEL,
            messages=valid_user_message,
            system=None,
            temperature=0.7,
            max_tokens=4096
        )
        # Check that leading/trailing whitespace is stripped
        assert result == "This is a full response."

@pytest.mark.asyncio
async def test_generate_full_response_no_content(valid_user_message): # <--- Add the fixture as an argument HERE
    """Should return an empty string if the API provides no content."""
    mock_api_response = MagicMock()
    mock_api_response.content = [] # Simulate an empty content list
    
    with patch(f"{CLAUDE_SERVICE_PATH}.async_client.messages.create", new=AsyncMock(return_value=mock_api_response)):
        # Now, valid_user_message is the list of dictionaries from the fixture
        result = await generate_full_response(valid_user_message) 
        assert result == ""