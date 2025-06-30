"""
Service interfaces for the application layer.

These interfaces define the contracts for application services that
orchestrate business logic and coordinate between different components.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

from ..entities.client import ClientSession
from ..entities.conversation import Conversation, Message
from ..entities.agent import Agent
from ..entities.analytics import SessionAnalytics, ConversationAnalytics, SystemMetrics


class IConnectionService(ABC):
    """Interface for client connection management."""
    
    @abstractmethod
    async def connect_client(
        self,
        client_id: str,
        websocket: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Connect a new client and return success status."""
        pass
    
    @abstractmethod
    async def disconnect_client(self, client_id: str) -> None:
        """Disconnect a client and cleanup resources."""
        pass
    
    @abstractmethod
    async def authenticate_client(
        self,
        client_id: str,
        api_key: str
    ) -> bool:
        """Authenticate a client with API key."""
        pass
    
    @abstractmethod
    async def get_connected_clients(self) -> List[str]:
        """Get list of currently connected client IDs."""
        pass
    
    @abstractmethod
    async def get_client_session(self, client_id: str) -> Optional[ClientSession]:
        """Get client session by ID."""
        pass
    
    @abstractmethod
    async def send_message_to_client(
        self,
        client_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """Send a message to a specific client."""
        pass
    
    @abstractmethod
    async def broadcast_message(
        self,
        message: Dict[str, Any],
        exclude_clients: Optional[List[str]] = None
    ) -> int:
        """Broadcast message to all connected clients."""
        pass
    
    @abstractmethod
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        pass


class IConversationService(ABC):
    """Interface for conversation management."""
    
    @abstractmethod
    async def start_conversation(
        self,
        client_id: str,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new conversation and return conversation ID."""
        pass
    
    @abstractmethod
    async def process_message(
        self,
        conversation_id: str,
        message_content: str,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a user message and return agent response."""
        pass
    
    @abstractmethod
    async def process_audio_message(
        self,
        conversation_id: str,
        audio_data: bytes,
        audio_format: str = "webm",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process an audio message and return agent response."""
        pass
    
    @abstractmethod
    async def end_conversation(
        self,
        conversation_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """End a conversation."""
        pass
    
    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        pass
    
    @abstractmethod
    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get conversation message history."""
        pass
    
    @abstractmethod
    async def update_conversation_context(
        self,
        conversation_id: str,
        context_updates: Dict[str, Any]
    ) -> bool:
        """Update conversation context."""
        pass
    
    @abstractmethod
    async def get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """Get AI-generated conversation summary."""
        pass


class IVoiceService(ABC):
    """Interface for voice call management."""
    
    @abstractmethod
    async def start_voice_call(
        self,
        client_id: str,
        conversation_id: Optional[str] = None
    ) -> bool:
        """Start a voice call session."""
        pass
    
    @abstractmethod
    async def end_voice_call(self, client_id: str) -> float:
        """End voice call and return duration in minutes."""
        pass
    
    @abstractmethod
    async def process_audio_chunk(
        self,
        client_id: str,
        audio_chunk: bytes,
        is_final: bool = False
    ) -> Optional[str]:
        """Process real-time audio chunk and return transcription."""
        pass
    
    @abstractmethod
    async def synthesize_speech(
        self,
        text: str,
        voice_config: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Convert text to speech audio."""
        pass
    
    @abstractmethod
    async def get_voice_call_status(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get current voice call status."""
        pass
    
    @abstractmethod
    async def configure_voice_settings(
        self,
        client_id: str,
        settings: Dict[str, Any]
    ) -> bool:
        """Configure voice settings for a client."""
        pass
    
    @abstractmethod
    async def get_supported_voices(self) -> List[Dict[str, Any]]:
        """Get list of supported voice models."""
        pass


class IAgentService(ABC):
    """Interface for agent management."""
    
    @abstractmethod
    async def create_agent(
        self,
        name: str,
        agent_type: str,
        config: Dict[str, Any],
        capabilities: Optional[List[str]] = None
    ) -> str:
        """Create a new agent and return agent ID."""
        pass
    
    @abstractmethod
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        pass
    
    @abstractmethod
    async def update_agent(
        self,
        agent_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update agent configuration."""
        pass
    
    @abstractmethod
    async def activate_agent(self, agent_id: str) -> bool:
        """Activate an agent."""
        pass
    
    @abstractmethod
    async def deactivate_agent(self, agent_id: str) -> bool:
        """Deactivate an agent."""
        pass
    
    @abstractmethod
    async def get_available_agent(
        self,
        capabilities: Optional[List[str]] = None,
        agent_type: Optional[str] = None
    ) -> Optional[Agent]:
        """Get an available agent that matches criteria."""
        pass
    
    @abstractmethod
    async def assign_agent_to_conversation(
        self,
        conversation_id: str,
        agent_id: Optional[str] = None
    ) -> Optional[str]:
        """Assign an agent to a conversation."""
        pass
    
    @abstractmethod
    async def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get agent performance metrics."""
        pass
    
    @abstractmethod
    async def scale_agents(
        self,
        target_count: int,
        agent_type: Optional[str] = None
    ) -> int:
        """Scale the number of active agents."""
        pass


class IAnalyticsService(ABC):
    """Interface for analytics and reporting."""
    
    @abstractmethod
    async def record_session_start(
        self,
        session_id: str,
        client_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record the start of a session."""
        pass
    
    @abstractmethod
    async def record_session_end(
        self,
        session_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Record the end of a session with metrics."""
        pass
    
    @abstractmethod
    async def record_conversation_metrics(
        self,
        conversation_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Record conversation-level metrics."""
        pass
    
    @abstractmethod
    async def get_real_time_metrics(self) -> SystemMetrics:
        """Get current real-time system metrics."""
        pass
    
    @abstractmethod
    async def get_dashboard_data(
        self,
        timeframe: str = "24h",
        company_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get dashboard analytics data."""
        pass
    
    @abstractmethod
    async def generate_report(
        self,
        report_type: str,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate an analytics report."""
        pass
    
    @abstractmethod
    async def get_performance_trends(
        self,
        metric: str,
        timeframe: str = "7d"
    ) -> List[Dict[str, Any]]:
        """Get performance trends for a metric."""
        pass
    
    @abstractmethod
    async def alert_on_metrics(
        self,
        thresholds: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check metrics against thresholds and return alerts."""
        pass


class INotificationService(ABC):
    """Interface for notification management."""
    
    @abstractmethod
    async def send_notification(
        self,
        recipient: str,
        message: str,
        notification_type: str = "info",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a notification to a recipient."""
        pass
    
    @abstractmethod
    async def send_email(
        self,
        to_address: str,
        subject: str,
        body: str,
        is_html: bool = False
    ) -> bool:
        """Send an email notification."""
        pass
    
    @abstractmethod
    async def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """Send a webhook notification."""
        pass
    
    @abstractmethod
    async def schedule_notification(
        self,
        recipient: str,
        message: str,
        send_at: datetime,
        notification_type: str = "info"
    ) -> str:
        """Schedule a notification for future delivery."""
        pass
    
    @abstractmethod
    async def cancel_scheduled_notification(self, notification_id: str) -> bool:
        """Cancel a scheduled notification."""
        pass


class IAuthenticationService(ABC):
    """Interface for authentication and authorization."""
    
    @abstractmethod
    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return company info."""
        pass
    
    @abstractmethod
    async def check_rate_limits(
        self,
        company_id: str,
        endpoint: str
    ) -> bool:
        """Check if company is within rate limits."""
        pass
    
    @abstractmethod
    async def increment_usage(
        self,
        company_id: str,
        resource_type: str,
        amount: int = 1
    ) -> None:
        """Increment usage counter for rate limiting."""
        pass
    
    @abstractmethod
    async def get_permissions(self, company_id: str) -> List[str]:
        """Get permissions for a company."""
        pass
    
    @abstractmethod
    async def has_permission(
        self,
        company_id: str,
        permission: str
    ) -> bool:
        """Check if company has a specific permission."""
        pass


class IHealthService(ABC):
    """Interface for health monitoring and diagnostics."""
    
    @abstractmethod
    async def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        pass
    
    @abstractmethod
    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service."""
        pass
    
    @abstractmethod
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics."""
        pass
    
    @abstractmethod
    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run system diagnostics."""
        pass
    
    @abstractmethod
    async def get_service_dependencies(self) -> Dict[str, List[str]]:
        """Get service dependency map."""
        pass


class IConfigurationService(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    async def get_config(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """Get configuration value."""
        pass
    
    @abstractmethod
    async def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        pass
    
    @abstractmethod
    async def reload_config(self) -> bool:
        """Reload configuration from source."""
        pass
    
    @abstractmethod
    async def validate_config(self) -> List[str]:
        """Validate configuration and return any issues."""
        pass
    
    @abstractmethod
    async def get_feature_flags(self) -> Dict[str, bool]:
        """Get current feature flag settings."""
        pass
    
    @abstractmethod
    async def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        pass