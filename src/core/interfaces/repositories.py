from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from core.entities.client import ClientSession
from core.entities.conversation import Conversation
from core.entities.analytics import SessionEvent, DailyMetrics

class IClientRepository(ABC):
    """Repository interface for client data persistence"""
    
    @abstractmethod
    async def record_connection(self, session: ClientSession) -> None:
        """Record a new client connection"""
        pass
    
    @abstractmethod
    async def record_disconnection(self, session: ClientSession) -> None:
        """Record client disconnection with session data"""
        pass
    
    @abstractmethod
    async def authenticate_company(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate company by API key"""
        pass
    
    @abstractmethod
    async def get_company_by_id(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Get company information by ID"""
        pass

class IConversationRepository(ABC):
    """Repository interface for conversation persistence"""
    
    @abstractmethod
    async def create_conversation(self, conversation: Conversation) -> str:
        """Create new conversation and return ID"""
        pass
    
    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        pass
    
    @abstractmethod
    async def update_conversation(self, conversation: Conversation) -> None:
        """Update existing conversation"""
        pass
    
    @abstractmethod
    async def get_or_create_conversation(
        self, 
        customer_id: str, 
        company_id: str, 
        agent_id: Optional[str] = None
    ) -> Conversation:
        """Get existing or create new conversation"""
        pass

class IAnalyticsRepository(ABC):
    """Repository interface for analytics data"""
    
    @abstractmethod
    async def record_event(self, event: SessionEvent) -> None:
        """Record analytics event"""
        pass
    
    @abstractmethod
    async def get_daily_metrics(self, company_id: str, date: date) -> Optional[DailyMetrics]:
        """Get daily metrics for company"""
        pass
    
    @abstractmethod
    async def update_daily_metrics(self, metrics: DailyMetrics) -> None:
        """Update daily metrics"""
        pass
    
    @abstractmethod
    async def get_company_usage_report(
        self, 
        company_id: str, 
        start_date: date, 
        end_date: date
    ) -> Dict[str, Any]:
        """Get usage report for date range"""
        pass

class IAgentRepository(ABC):
    """Repository interface for agent data"""
    
    @abstractmethod
    async def get_agent(self, agent_id: str, company_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information"""
        pass
    
    @abstractmethod
    async def get_company_agents(self, company_id: str) -> List[Dict[str, Any]]:
        """Get all agents for company"""
        pass
    
    @abstractmethod
    async def get_base_agent(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Get base agent for company"""
        pass
