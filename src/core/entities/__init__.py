"""
Core business entities for the AI voice calling system.

This module contains domain entities that represent the core business concepts:
- ClientSession: Represents a connected client with their state and metadata
- Conversation: Represents a conversation thread with messages and context
- Agent: Represents an AI agent with its configuration and capabilities
- Analytics: Represents metrics and analytics data

These entities contain business logic and are independent of any framework
or infrastructure concerns.
"""

from .client import (
    ClientSession,
    ConnectionState,
    VoiceCallState,
    ClientMetrics
)
from .conversation import (
    Conversation,
    Message,
    MessageType,
    ConversationState,
    ConversationMetrics
)
from .agent import (
    Agent,
    AgentType,
    AgentCapability,
    AgentConfig,
    AgentMetrics
)
from .analytics import (
    SessionAnalytics,
    ConversationAnalytics,
    SystemMetrics,
    PerformanceMetric,
    AnalyticsTimeframe
)

__all__ = [
    # Client entities
    "ClientSession",
    "ConnectionState", 
    "VoiceCallState",
    "ClientMetrics",
    
    # Conversation entities
    "Conversation",
    "Message",
    "MessageType",
    "ConversationState",
    "ConversationMetrics",
    
    # Agent entities
    "Agent",
    "AgentType",
    "AgentCapability", 
    "AgentConfig",
    "AgentMetrics",
    
    # Analytics entities
    "SessionAnalytics",
    "ConversationAnalytics",
    "SystemMetrics",
    "PerformanceMetric",
    "AnalyticsTimeframe"
]