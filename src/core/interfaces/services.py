# src/core/interfaces/services.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncGenerator, Callable
from fastapi import WebSocket
from core.entities.client import ClientSession
from core.entities.conversation import Conversation
from core.entities.analytics import LiveStats, DailyMetrics

class IConnectionService(ABC):
    """Service interface for managing client connections"""
    
    @abstractmethod
    async def connect_client(self, client_id: str, websocket: WebSocket) -> bool:
        """Accept and establish client connection"""
        pass
    
    @abstractmethod
    async def disconnect_client(self, client_id: str) -> None:
        """Disconnect client and cleanup resources"""
        pass
    
    @abstractmethod
    async def authenticate_client(self, client_id: str, api_key: str) -> bool:
        """Authenticate client with company API key"""
        pass
    
    @abstractmethod
    async def initialize_agent(self, client_id: str, agent_id: Optional[str] = None) -> bool:
        """Initialize AI agent resources for client"""
        pass
    
    @abstractmethod
    def get_client_session(self, client_id: str) -> Optional[ClientSession]:
        """Get client session by ID"""
        pass
    
    @abstractmethod
    def get_active_clients(self) -> List[str]:
        """Get list of active client IDs"""
        pass
    
    @abstractmethod
    async def force_disconnect_client(self, client_id: str, reason: str = "Admin disconnect") -> bool:
        """Forcefully disconnect a client"""
        pass

class IConversationService(ABC):
    """Service interface for conversation management"""
    
    @abstractmethod
    async def process_message(
        self, 
        client_id: str, 
        message: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Process user message and return streaming AI response"""
        pass
    
    @abstractmethod
    async def get_conversation_context(
        self, 
        conversation_id: str, 
        max_messages: int = 5
    ) -> List[Dict[str, str]]:
        """Get conversation context for AI"""
        pass
    
    @abstractmethod
    async def create_conversation(
        self, 
        customer_id: str, 
        company_id: str, 
        agent_id: Optional[str] = None
    ) -> str:
        """Create new conversation and return ID"""
        pass
    
    @abstractmethod
    async def end_conversation(self, conversation_id: str) -> None:
        """End conversation and update final metrics"""
        pass

class IVoiceService(ABC):
    """Service interface for voice call management"""
    
    @abstractmethod
    async def start_voice_call(
        self, 
        client_id: str, 
        voice_callback: Optional[Callable] = None
    ) -> bool:
        """Start voice call session"""
        pass
    
    @abstractmethod
    async def end_voice_call(self, client_id: str) -> float:
        """End voice call and return duration"""
        pass
    
    @abstractmethod
    async def process_audio_chunk(self, client_id: str, audio_data: bytes) -> None:
        """Process incoming audio data"""
        pass
    
    @abstractmethod
    async def synthesize_speech(self, client_id: str, text: str) -> bytes:
        """Convert text to speech for client"""
        pass

class IAnalyticsService(ABC):
    """Service interface for analytics and metrics"""
    
    @abstractmethod
    async def record_connection(self, session: ClientSession) -> None:
        """Record client connection event"""
        pass
    
    @abstractmethod
    async def record_disconnection(self, session: ClientSession) -> None:
        """Record client disconnection with session data"""
        pass
    
    @abstractmethod
    async def record_message(
        self, 
        client_id: str, 
        tokens: int, 
        response_time: float
    ) -> None:
        """Record message processing metrics"""
        pass
    
    @abstractmethod
    async def record_voice_call(self, client_id: str, duration: float) -> None:
        """Record voice call completion"""
        pass
    
    @abstractmethod
    async def record_error(self, company_id: str, error_message: str) -> None:
        """Record error occurrence"""
        pass
    
    @abstractmethod
    async def get_live_stats(self) -> LiveStats:
        """Get current live system statistics"""
        pass
    
    @abstractmethod
    async def get_company_usage_report(
        self, 
        company_id: str, 
        start_date: str, 
        end_date: str
    ) -> Dict[str, Any]:
        """Get usage report for company"""
        pass

class IAgentService(ABC):
    """Service interface for AI agent management"""
    
    @abstractmethod
    async def initialize_agent_resources(
        self, 
        client_id: str, 
        company_id: str, 
        agent_id: Optional[str] = None
    ) -> bool:
        """Initialize agent resources (RAG, prompts, etc.)"""
        pass
    
    @abstractmethod
    async def get_best_agent(
        self, 
        company_id: str, 
        query: str, 
        current_agent_id: Optional[str] = None
    ) -> Optional[str]:
        """Find best agent for given query"""
        pass
    
    @abstractmethod
    async def cleanup_agent_resources(self, client_id: str) -> None:
        """Cleanup agent resources for client"""
        pass

class INotificationService(ABC):
    """Service interface for notifications and messaging"""
    
    @abstractmethod
    async def send_to_client(
        self, 
        client_id: str, 
        message: Dict[str, Any]
    ) -> bool:
        """Send message to specific client"""
        pass
    
    @abstractmethod
    async def broadcast_to_company(
        self, 
        company_id: str, 
        message: Dict[str, Any],
        exclude_client: Optional[str] = None
    ) -> int:
        """Broadcast message to all clients of a company"""
        pass
    
    @abstractmethod
    async def send_error(self, client_id: str, error_message: str) -> None:
        """Send error message to client"""
        pass

class IHealthService(ABC):
    """Service interface for system health monitoring"""
    
    @abstractmethod
    async def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        pass
    
    @abstractmethod
    async def check_service_health(self, service_name: str) -> bool:
        """Check health of specific service"""
        pass
    
    @abstractmethod
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        pass