# src/api/claude/__init__.py

# Import the functions that actually exist in your service file.
from .service import (
    stream_text_response,
    generate_full_response
)

# Import utility functions if they are used elsewhere.
from .utils import (
    validate_claude_message_format,
    construct_claude_developer_message,
    construct_claude_user_message,
    construct_claude_assistant_message,
    log_and_upload_to_s3,
    split_text_into_chunks
)

# Update the __all__ list to only export existing functions.
__all__ = [
    "stream_text_response",
    "generate_full_response",
    "validate_claude_message_format",
    "construct_claude_developer_message",
    "construct_claude_user_message",
    "construct_claude_assistant_message",
    "log_and_upload_to_s3",
    "split_text_into_chunks"
]