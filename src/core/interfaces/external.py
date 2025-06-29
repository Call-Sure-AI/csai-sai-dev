# src/core/interfaces/external.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator

class ISpeechToTextService(ABC):
    """Interface for speech-to-text services"""
    
    @abstractmethod
    async def initialize_session(self, session_id: str, callback: Callable) -> bool:
        """Initialize STT session with callback"""
        pass
    
    @abstractmethod
    async def process_audio_chunk(self, session_id: str, audio_data: bytes) -> None:
        """Process audio chunk"""
        pass
    
    @abstractmethod
    async def close_session(self, session_id: str) -> None:
        """Close STT session"""
        pass

class ITextToSpeechService(ABC):
    """Interface for text-to-speech services"""
    
    @abstractmethod
    async def generate_audio(self, text: str, voice_settings: Optional[Dict] = None) -> bytes:
        """Generate audio from text"""
        pass
    
    @abstractmethod
    async def stream_audio(
        self, 
        text: str, 
        callback: Callable,
        voice_settings: Optional[Dict] = None
    ) -> None:
        """Stream audio generation"""
        pass

class IAIService(ABC):
    """Interface for AI/LLM services"""
    
    @abstractmethod
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        agent_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming AI response"""
        pass
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate text embedding"""
        pass

class IVectorStoreService(ABC):
    """Interface for vector store operations"""
    
    @abstractmethod
    async def search(
        self, 
        company_id: str, 
        query_embedding: List[float],
        agent_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search vector store"""
        pass
    
    @abstractmethod
    async def add_documents(
        self, 
        company_id: str, 
        agent_id: str,
        documents: List[Dict[str, Any]]
    ) -> bool:
        """Add documents to vector store"""
        pass
    
    @abstractmethod
    async def delete_agent_data(self, company_id: str, agent_id: str) -> bool:
        """Delete agent data from vector store"""
        pass

# src/core/interfaces/events.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Callable
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DomainEvent:
    """Base domain event"""
    event_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]

class IEventPublisher(ABC):
    """Interface for publishing domain events"""
    
    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish domain event"""
        pass

class IEventSubscriber(ABC):
    """Interface for subscribing to domain events"""
    
    @abstractmethod
    async def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to event type"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from event type"""
        pass

class IEventStore(ABC):
    """Interface for event storage"""
    
    @abstractmethod
    async def append_event(self, stream_id: str, event: DomainEvent) -> None:
        """Append event to stream"""
        pass
    
    @abstractmethod
    async def get_events(self, stream_id: str, from_version: int = 0) -> List[DomainEvent]:
        """Get events from stream"""
        pass