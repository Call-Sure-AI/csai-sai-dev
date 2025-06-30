"""
Repository interfaces for data access layer.

These interfaces define the contracts for data persistence and retrieval
operations. They are implemented by the infrastructure layer and used
by the application services.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..entities.client import ClientSession
from ..entities.conversation import Conversation, Message
from ..entities.agent import Agent
from ..entities.analytics import SessionAnalytics, ConversationAnalytics, SystemMetrics


class IClientRepository(ABC):
    """Interface for client session data access."""
    
    @abstractmethod
    async def save_session(self, session: ClientSession) -> None:
        """Save or update a client session."""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[ClientSession]:
        """Get a client session by ID."""
        pass
    
    @abstractmethod
    async def get_sessions_by_client(self, client_id: str) -> List[ClientSession]:
        """Get all sessions for a client."""
        pass
    
    @abstractmethod
    async def get_active_sessions(self) -> List[ClientSession]:
        """Get all currently active sessions."""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        pass
    
    @abstractmethod
    async def authenticate_company(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate a company by API key."""
        pass
    
    @abstractmethod
    async def record_connection_event(
        self,
        session_id: str,
        event_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a connection event."""
        pass
    
    @abstractmethod
    async def get_session_count_by_company(self, company_id: str) -> int:
        """Get active session count for a company."""
        pass


class IConversationRepository(ABC):
    """Interface for conversation data access."""
    
    @abstractmethod
    async def save_conversation(self, conversation: Conversation) -> None:
        """Save or update a conversation."""
        pass
    
    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        pass
    
    @abstractmethod
    async def get_conversations_by_client(self, client_id: str) -> List[Conversation]:
        """Get all conversations for a client."""
        pass
    
    @abstractmethod
    async def get_conversations_by_agent(self, agent_id: str) -> List[Conversation]:
        """Get all conversations handled by an agent."""
        pass
    
    @abstractmethod
    async def save_message(self, message: Message) -> None:
        """Save a message."""
        pass
    
    @abstractmethod
    async def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Message]:
        """Get messages for a conversation."""
        pass
    
    @abstractmethod
    async def get_recent_conversations(
        self,
        limit: int = 10,
        company_id: Optional[str] = None
    ) -> List[Conversation]:
        """Get recent conversations."""
        pass
    
    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages."""
        pass
    
    @abstractmethod
    async def search_conversations(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Conversation]:
        """Search conversations by content or metadata."""
        pass


class IAgentRepository(ABC):
    """Interface for agent data access."""
    
    @abstractmethod
    async def save_agent(self, agent: Agent) -> None:
        """Save or update an agent."""
        pass
    
    @abstractmethod
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        pass
    
    @abstractmethod
    async def get_all_agents(self) -> List[Agent]:
        """Get all agents."""
        pass
    
    @abstractmethod
    async def get_active_agents(self) -> List[Agent]:
        """Get all active agents."""
        pass
    
    @abstractmethod
    async def get_available_agents(self) -> List[Agent]:
        """Get all available agents that can handle new conversations."""
        pass
    
    @abstractmethod
    async def get_agents_by_capability(self, capability: str) -> List[Agent]:
        """Get agents that have a specific capability."""
        pass
    
    @abstractmethod
    async def get_agents_by_type(self, agent_type: str) -> List[Agent]:
        """Get agents of a specific type."""
        pass
    
    @abstractmethod
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        pass
    
    @abstractmethod
    async def update_agent_load(self, agent_id: str, current_load: int) -> None:
        """Update an agent's current load."""
        pass
    
    @abstractmethod
    async def get_agent_performance_history(
        self,
        agent_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get agent performance history."""
        pass


class IAnalyticsRepository(ABC):
    """Interface for analytics data access."""
    
    @abstractmethod
    async def save_session_analytics(self, analytics: SessionAnalytics) -> None:
        """Save session analytics data."""
        pass
    
    @abstractmethod
    async def save_conversation_analytics(self, analytics: ConversationAnalytics) -> None:
        """Save conversation analytics data."""
        pass
    
    @abstractmethod
    async def save_system_metrics(self, metrics: SystemMetrics) -> None:
        """Save system metrics."""
        pass
    
    @abstractmethod
    async def get_session_analytics(self, session_id: str) -> Optional[SessionAnalytics]:
        """Get session analytics by session ID."""
        pass
    
    @abstractmethod
    async def get_conversation_analytics(self, conversation_id: str) -> Optional[ConversationAnalytics]:
        """Get conversation analytics by conversation ID."""
        pass
    
    @abstractmethod
    async def get_system_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        timeframe: str = "hourly"
    ) -> List[SystemMetrics]:
        """Get system metrics for a time range."""
        pass
    
    @abstractmethod
    async def get_company_analytics(
        self,
        company_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get aggregated analytics for a company."""
        pass
    
    @abstractmethod
    async def get_agent_analytics(
        self,
        agent_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get aggregated analytics for an agent."""
        pass
    
    @abstractmethod
    async def get_performance_trends(
        self,
        metric_type: str,
        start_time: datetime,
        end_time: datetime,
        granularity: str = "hourly"
    ) -> List[Dict[str, Any]]:
        """Get performance trends for a specific metric."""
        pass
    
    @abstractmethod
    async def get_top_performers(
        self,
        metric_type: str,
        limit: int = 10,
        time_period: str = "24h"
    ) -> List[Dict[str, Any]]:
        """Get top performing entities for a metric."""
        pass
    
    @abstractmethod
    async def aggregate_metrics(
        self,
        filters: Dict[str, Any],
        group_by: List[str],
        metrics: List[str]
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics with custom filters and grouping."""
        pass
    
    @abstractmethod
    async def cleanup_old_analytics(self, retention_days: int = 90) -> int:
        """Clean up old analytics data and return number of records deleted."""
        pass


class ICompanyRepository(ABC):
    """Interface for company/organization data access."""
    
    @abstractmethod
    async def get_company(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Get company details by ID."""
        pass
    
    @abstractmethod
    async def get_company_by_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get company details by API key."""
        pass
    
    @abstractmethod
    async def save_company(self, company_data: Dict[str, Any]) -> str:
        """Save company data and return company ID."""
        pass
    
    @abstractmethod
    async def update_company(self, company_id: str, updates: Dict[str, Any]) -> bool:
        """Update company data."""
        pass
    
    @abstractmethod
    async def get_company_usage(self, company_id: str) -> Dict[str, Any]:
        """Get company usage statistics."""
        pass
    
    @abstractmethod
    async def get_company_limits(self, company_id: str) -> Dict[str, Any]:
        """Get company rate limits and quotas."""
        pass
    
    @abstractmethod
    async def increment_usage(
        self,
        company_id: str,
        metric: str,
        amount: int = 1
    ) -> None:
        """Increment usage counter for a company."""
        pass


class IConfigurationRepository(ABC):
    """Interface for system configuration data access."""
    
    @abstractmethod
    async def get_config(self, key: str) -> Optional[Any]:
        """Get configuration value by key."""
        pass
    
    @abstractmethod
    async def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        pass
    
    @abstractmethod
    async def get_all_configs(self) -> Dict[str, Any]:
        """Get all configuration values."""
        pass
    
    @abstractmethod
    async def delete_config(self, key: str) -> bool:
        """Delete a configuration key."""
        pass
    
    @abstractmethod
    async def get_configs_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """Get all configuration values with a specific prefix."""
        pass


class IHealthCheckRepository(ABC):
    """Interface for health check and monitoring data access."""
    
    @abstractmethod
    async def save_health_check(
        self,
        service_name: str,
        status: str,
        details: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Save health check result."""
        pass
    
    @abstractmethod
    async def get_service_health(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get latest health status for a service."""
        pass
    
    @abstractmethod
    async def get_all_service_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all services."""
        pass
    
    @abstractmethod
    async def get_health_history(
        self,
        service_name: str,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get health check history for a service."""
        pass
    
    @abstractmethod
    async def cleanup_old_health_data(self, retention_hours: int = 168) -> int:
        """Clean up old health check data (default 7 days)."""
        pass


class IAuditRepository(ABC):
    """Interface for audit log data access."""
    
    @abstractmethod
    async def log_event(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        user_id: Optional[str],
        details: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log an audit event."""
        pass
    
    @abstractmethod
    async def get_audit_logs(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit logs with filters."""
        pass
    
    @abstractmethod
    async def get_entity_history(
        self,
        entity_type: str,
        entity_id: str
    ) -> List[Dict[str, Any]]:
        """Get change history for a specific entity."""
        pass
    
    @abstractmethod
    async def cleanup_old_audit_logs(self, retention_days: int = 365) -> int:
        """Clean up old audit logs."""
        pass