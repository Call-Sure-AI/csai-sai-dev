from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, Any, Optional
from enum import Enum

class EventType(Enum):
    CONNECTION = "connection"
    DISCONNECTION = "disconnection"
    MESSAGE = "message"
    VOICE_START = "voice_start"
    VOICE_END = "voice_end"
    ERROR = "error"
    TICKET_CREATED = "ticket_created"

@dataclass
class SessionEvent:
    """Analytics event for session tracking"""
    id: str
    company_id: str
    client_id: str
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Event-specific data
    session_duration: Optional[float] = None
    message_count: Optional[int] = None
    token_count: Optional[int] = None
    voice_duration: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DailyMetrics:
    """Daily aggregated metrics for analytics"""
    company_id: str
    date: date
    
    # Connection metrics
    total_connections: int = 0
    peak_concurrent_connections: int = 0
    connection_hours: float = 0.0
    
    # Message metrics
    total_messages: int = 0
    total_tokens: int = 0
    
    # Voice metrics
    total_voice_calls: int = 0
    total_voice_minutes: float = 0.0
    
    # Business metrics
    tickets_created: int = 0
    errors_count: int = 0
    avg_response_time: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_metrics(self, **kwargs) -> None:
        """Update metrics with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                if key == "avg_response_time":
                    # Calculate running average
                    current_messages = self.total_messages or 1
                    current_avg = getattr(self, key)
                    setattr(self, key, (current_avg * (current_messages - 1) + value) / current_messages)
                else:
                    # Add to existing value
                    current_value = getattr(self, key)
                    setattr(self, key, current_value + value)
        
        self.updated_at = datetime.utcnow()

@dataclass
class LiveStats:
    """Live system statistics"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Connection stats
    total_connections: int = 0
    initialized_connections: int = 0
    voice_calls_active: int = 0
    companies_active: int = 0
    
    # Performance stats
    processing_active: int = 0
    processing_capacity: int = 20
    memory_usage_mb: float = 0.0
    
    # Rate limiting
    rate_limited_clients: int = 0
    
    def calculate_utilization(self, max_connections: int) -> float:
        """Calculate connection utilization percentage"""
        return (self.total_connections / max_connections) * 100 if max_connections > 0 else 0
    
    def calculate_processing_utilization(self) -> float:
        """Calculate processing utilization percentage"""
        return (self.processing_active / self.processing_capacity) * 100