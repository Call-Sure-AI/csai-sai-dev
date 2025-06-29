# src/api/claude/service.py

from anthropic import AsyncAnthropic, APIStatusError
from typing import AsyncGenerator, List, Dict, Optional, Tuple
import asyncio
import logging
from src.config.settings import CLAUDE_API_KEY

# Using the latest Sonnet 4 model, which is ideal for low-latency voice apps.
DEFAULT_CLAUDE_MODEL = "claude-4-sonnet-20250522"

# Initialize Anthropic async client with a reasonable timeout
async_client = AsyncAnthropic(api_key=CLAUDE_API_KEY, timeout=60.0)
logger = logging.getLogger(__name__)


def _prepare_claude_messages(messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """
    Validates and prepares messages for the Anthropic API.
    - Extracts the 'system' prompt.
    - Ensures the conversation starts with a 'user' message.
    """
    if not messages:
        raise ValueError("The 'messages' list cannot be empty.")

    system_prompt = None
    conversation_messages = messages

    if messages and messages[0]['role'] == 'system':
        system_prompt = messages[0]['content']
        conversation_messages = messages[1:]
    
    if not conversation_messages or conversation_messages[0]['role'] != 'user':
        raise ValueError("The first message after an optional system prompt must be from the 'user'.")

    return system_prompt, conversation_messages


async def stream_text_response(messages: List[dict], model: str = DEFAULT_CLAUDE_MODEL, temperature: float = 0.7) -> AsyncGenerator[str, None]:
    """
    Stream a conversational text response from Claude in real-time.
    This implementation is diligently aligned with Anthropic's official streaming documentation.
    """
    try:
        system_prompt, conversation_messages = _prepare_claude_messages(messages)
        
        # Use the .stream() method to get an event-based stream
        async with async_client.messages.stream(
            model=model,
            messages=conversation_messages,
            system=system_prompt,
            temperature=temperature,
            max_tokens=4096
        ) as stream:
            # Iterate through the events in the stream
            async for event in stream:
                # Check for the content_block_delta event, which contains text chunks
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield event.delta.text
                
                # You can also handle other events for more advanced logic
                elif event.type == "message_start":
                    logger.debug(f"Stream started for model: {event.message.model}")
                
                elif event.type == "message_delta":
                    # This event contains metadata like the stop_reason
                    if event.delta.stop_reason:
                        logger.info(f"Stream finished with reason: {event.delta.stop_reason}")

                elif event.type == "message_stop":
                    logger.info("Stream processing complete.")

    except APIStatusError as e:
        logger.error(f"Anthropic API error during streaming: {e.status_code} - {e.response}", exc_info=True)
        raise RuntimeError(f"Anthropic API error: {e.message}") from e
    except Exception as e:
        logger.error(f"Error during Claude streaming: {e}", exc_info=True)
        raise RuntimeError(f"An unexpected error occurred during Claude streaming: {e}") from e


async def generate_full_response(messages: List[dict], model: str = DEFAULT_CLAUDE_MODEL, temperature: float = 0.7, max_tokens: int = 4096) -> str:
    """
    Generate a complete conversational response from Claude in one API call.
    This function remains unchanged as it does not involve event-based streaming.
    """
    try:
        system_prompt, conversation_messages = _prepare_claude_messages(messages)
        
        response = await async_client.messages.create(
            model=model,
            messages=conversation_messages,
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if response.content:
            return response.content[0].text.strip()
        return ""
        
    except APIStatusError as e:
        logger.error(f"Anthropic API error generating full response: {e.status_code} - {e.response}", exc_info=True)
        raise RuntimeError(f"Anthropic API error: {e.message}") from e
    except Exception as e:
        logger.error(f"Error generating full Claude response: {e}", exc_info=True)
        raise RuntimeError(f"An unexpected error occurred generating a full Claude response: {e}") from e