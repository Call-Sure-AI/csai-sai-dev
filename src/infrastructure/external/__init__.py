# src/infrastructure/external/__init__.py
"""
External service integrations for the AI voice calling system.

This module contains concrete implementations of external service interfaces
for AI models, speech services, and other third-party APIs.
"""

from .openai_service import OpenAIService
from .deepgram_service import DeepgramService
from .elevenlabs_service import ElevenLabsService

__all__ = [
    "OpenAIService",
    "DeepgramService", 
    "ElevenLabsService"
]