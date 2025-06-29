from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
import time

class ConnectionState(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    INITIALIZED = "initialized"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"

class VoiceCallState(Enum):
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    ENDING = "ending"

@dataclass
class ClientSession:
    """Core client session entity representing a connected client"""
    
    # Identity
    client_id: str
    websocket: Any  # WebSocket type
    
    # State
    state: ConnectionState = ConnectionState.CONNECTING
    connection_time: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    # Business context
    company: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_resources: Optional[Dict[str, Any]] = None
    
    # Session metrics
    message_count: int = 0
    total_tokens: int = 0
    request_times: List[float] = field(default_factory=list)
    error_count: int = 0
    
    # Voice call state
    voice_state: VoiceCallState = VoiceCallState.INACTIVE
    voice_start_time: Optional[datetime] = None
    voice_callback: Optional[Callable] = None
    
    # Rate limiting
    max_requests_per_minute: int = 60
    rate_limit_window: int = 60
    
    def update_activity(self, tokens: int = 0) -> None:
        """Update session activity and metrics"""
        self.last_activity = datetime.utcnow()
        self.message_count += 1
        self.total_tokens += tokens
    
    def set_company(self, company_data: Dict[str, Any]) -> None:
        """Set company information"""
        self.company = company_data
        self.state = ConnectionState.AUTHENTICATED
    
    def set_conversation(self, conversation_id: str) -> None:
        """Set conversation context"""
        self.conversation_id = conversation_id
    
    def set_agent_resources(self, agent_id: str, resources: Dict[str, Any]) -> None:
        """Set agent context and mark as initialized"""
        self.agent_id = agent_id
        self.agent_resources = resources
        self.state = ConnectionState.INITIALIZED
    
    def start_voice_call(self, callback: Optional[Callable] = None) -> bool:
        """Start voice call session"""
        if self.voice_state != VoiceCallState.INACTIVE:
            return False
        
        self.voice_state = VoiceCallState.ACTIVE
        self.voice_start_time = datetime.utcnow()
        self.voice_callback = callback
        return True
    
    def end_voice_call(self) -> float:
        """End voice call and return duration"""
        if self.voice_state != VoiceCallState.ACTIVE or not self.voice_start_time:
            return 0.0
        
        duration = (datetime.utcnow() - self.voice_start_time).total_seconds()
        self.voice_state = VoiceCallState.INACTIVE
        self.voice_start_time = None
        self.voice_callback = None
        return duration
    
    def check_rate_limit(self) -> bool:
        """Check if client is within rate limits"""
        now = time.time()
        cutoff = now - self.rate_limit_window
        
        # Clean old requests
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        if len(self.request_times) >= self.max_requests_per_minute:
            return False
        
        self.request_times.append(now)
        return True
    
    def get_session_duration(self) -> float:
        """Get session duration in seconds"""
        return (datetime.utcnow() - self.connection_time).total_seconds()
    
    def get_voice_duration(self) -> float:
        """Get current voice call duration"""
        if self.voice_start_time and self.voice_state == VoiceCallState.ACTIVE:
            return (datetime.utcnow() - self.voice_start_time).total_seconds()
        return 0.0
    
    def is_stale(self, stale_threshold_minutes: int = 30) -> bool:
        """Check if session is stale (inactive)"""
        threshold = datetime.utcnow() - timedelta(minutes=stale_threshold_minutes)
        return self.last_activity < threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "client_id": self.client_id,
            "state": self.state.value,
            "connection_time": self.connection_time.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "session_duration": self.get_session_duration(),
            "company": self.company,
            "conversation_id": self.conversation_id,
            "agent_id": self.agent_id,
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "voice_state": self.voice_state.value,
            "voice_duration": self.get_voice_duration(),
            "rate_limit_remaining": max(0, self.max_requests_per_minute - len(self.request_times))
        }