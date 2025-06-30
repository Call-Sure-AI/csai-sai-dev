"""
Core Domain Layer

This module contains the core business entities, interfaces, and domain logic
for the AI voice calling system. It is framework-agnostic and contains no
dependencies on external libraries or infrastructure concerns.

The core layer is organized into:
- entities: Business entities and value objects
- interfaces: Abstract interfaces and protocols
- exceptions: Domain-specific exceptions
"""

from .exceptions import (
    DomainException,
    ClientNotFoundException,
    ConversationNotFoundException,
    AgentNotFoundException,
    InvalidClientStateException,
    MaxConnectionsExceededException,
    AuthenticationFailedException,
    VoiceCallException,
    AnalyticsException,
    RateLimitExceededException,
    InvalidMessageFormatException,
    ResourceExhaustedException
)

__version__ = "1.0.0"
__all__ = [
    "DomainException",
    "ClientNotFoundException", 
    "ConversationNotFoundException",
    "AgentNotFoundException",
    "InvalidClientStateException",
    "MaxConnectionsExceededException",
    "AuthenticationFailedException",
    "VoiceCallException",
    "AnalyticsException",
    "RateLimitExceededException",
    "InvalidMessageFormatException",
    "ResourceExhaustedException"
]