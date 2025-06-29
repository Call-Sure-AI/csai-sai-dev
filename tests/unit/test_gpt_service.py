import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict

# Path to the module we are testing
GPT_SERVICE_PATH = "src.api.gpt.service"

from src.api.gpt.service import (
    stream_text_response,
    generate_full_response,
    generate_embedding,
    DEFAULT_GPT_MODEL
)
from src.api.gpt.utils import validate_gpt_message_format

# --- Test Data ---

@pytest.fixture
def valid_gpt_messages() -> List[Dict[str, str]]:
    """A valid message list for GPT, including a system role."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, GPT!"}
    ]

# --- Tests for GPT Chat ---

@pytest.mark.asyncio
async def test_gpt_stream_response_success(valid_gpt_messages):
    """Should stream text chunks successfully from GPT."""
    
    # This helper class simulates the structure of an OpenAI stream chunk
    class MockChoice:
        def __init__(self, content):
            self.delta = MagicMock()
            self.delta.content = content

    # The list of chunks our mock stream will produce
    mock_chunks = [
        MagicMock(choices=[MockChoice("Hello")]),
        MagicMock(choices=[MockChoice(", ")]),
        MagicMock(choices=[MockChoice("GPT!")]),
        MagicMock(choices=[MockChoice(None)])  # Represents the end of the stream
    ]

    # An async generator to yield the chunks, simulating a real stream
    async def chunk_generator():
        for chunk in mock_chunks:
            yield chunk

    # **THE FIX IS HERE**: We now use `new=AsyncMock(...)`.
    # This correctly mocks the awaitable `create` method so that it returns our async generator.
    with patch(f"{GPT_SERVICE_PATH}.async_client.chat.completions.create", new=AsyncMock(return_value=chunk_generator())) as mock_api_call:
        result_chunks = [chunk async for chunk in stream_text_response(valid_gpt_messages)]
        
        # Assertions
        mock_api_call.assert_awaited_once()
        assert "".join(result_chunks) == "Hello, GPT!"


@pytest.mark.asyncio
async def test_gpt_full_response_success(valid_gpt_messages):
    """Should return a complete text response from GPT."""
    
    # **THE FIX IS HERE**: The application code expects a dictionary-like object for `message`.
    # We set up the mock to return a dictionary for the `message` attribute.
    mock_message = {"content": " This is a full response from GPT. "}
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = mock_message
    
    with patch(f"{GPT_SERVICE_PATH}.async_client.chat.completions.create", new=AsyncMock(return_value=mock_response)) as mock_api_call:
        result = await generate_full_response(valid_gpt_messages)
        
        mock_api_call.assert_awaited_once()
        # The .strip() is called in the application code, so we assert against the stripped version.
        assert result == "This is a full response from GPT."


# --- Tests for GPT Embeddings ---

@pytest.mark.asyncio
async def test_gpt_generate_embedding_success():
    """Should successfully generate an embedding vector."""
    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1, 0.2, 0.3, 0.4]
    
    mock_response = MagicMock()
    mock_response.data = [mock_embedding_data]
    
    with patch(f"{GPT_SERVICE_PATH}.async_client.embeddings.create", new=AsyncMock(return_value=mock_response)) as mock_api_call:
        embedding = await generate_embedding("test input")
        
        mock_api_call.assert_awaited_once_with(
            input="test input",
            model="text-embedding-3-small"
        )
        assert embedding == [0.1, 0.2, 0.3, 0.4]