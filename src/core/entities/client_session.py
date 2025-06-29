from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from fastapi import WebSocket

@dataclass
class ClientSession:
    """Extracted and cleaned up from your existing ClientSession class"""
    
    # Core identification
    client_id: str
    websocket: WebSocket
    
    # Connection state
    connected: bool = True
    initialized: bool = False
    connection_time: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    # Business context
    company: Optional[Dict[str, Any]] = None
    conversation: Optional[Any] = None  # Will be Conversation entity
    agent_id: Optional[str] = None
    agent_resources: Optional[Dict[str, Any]] = None
    
    # Session metrics
    message_count: int = 0
    total_tokens: int = 0
    request_times: List[float] = field(default_factory=list)
    
    # Voice call state
    is_voice_call: bool = False
    voice_start_time: Optional[datetime] = None
    voice_callback: Optional[Callable] = None
    
    def update_activity(self, tokens: int = 0) -> None:
        """Update session activity metrics"""
        self.last_activity = datetime.utcnow()
        self.message_count += 1
        self.total_tokens += tokens
    
    def set_company(self, company_data: Dict[str, Any]) -> None:
        """Set company information"""
        self.company = company_data
    
    def set_conversation(self, conversation: Any) -> None:
        """Set conversation context"""
        self.conversation = conversation
    
    def set_agent_resources(self, agent_id: str, resources: Dict[str, Any]) -> None:
        """Set agent context and mark as initialized"""
        self.agent_id = agent_id
        self.agent_resources = resources
        self.initialized = True
    
    def start_voice_call(self, voice_callback: Optional[Callable] = None) -> None:
        """Start voice call session"""
        self.is_voice_call = True
        self.voice_start_time = datetime.utcnow()
        self.voice_callback = voice_callback
    
    def end_voice_call(self) -> float:
        """End voice call and return duration"""
        if not self.is_voice_call or not self.voice_start_time:
            return 0.0
        
        duration = (datetime.utcnow() - self.voice_start_time).total_seconds()
        self.is_voice_call = False
        self.voice_start_time = None
        self.voice_callback = None
        return duration
    
    def check_rate_limit(self, max_requests: int, window_seconds: int) -> bool:
        """Check if client is within rate limits"""
        import time
        now = time.time()
        cutoff = now - window_seconds
        
        # Clean old requests
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        if len(self.request_times) >= max_requests:
            return False
        
        self.request_times.append(now)
        return True
    
    def get_session_duration(self) -> float:
        """Get session duration in seconds"""
        return (datetime.utcnow() - self.connection_time).total_seconds()
    
    def get_voice_duration(self) -> float:
        """Get current voice call duration"""
        if self.voice_start_time and self.is_voice_call:
            return (datetime.utcnow() - self.voice_start_time).total_seconds()
        return 0.0
    
    def is_websocket_closed(self) -> bool:
        """Check if WebSocket is closed"""
        try:
            if not getattr(self.websocket, '_peer_connected', True):
                return True
            client_state = getattr(self.websocket, 'client_state', None)
            if client_state and hasattr(client_state, 'name'):
                return client_state.name != 'CONNECTED'
            return getattr(self.websocket, '_closed', False)
        except Exception:
            return False
    
    def close_websocket(self) -> None:
        """Close WebSocket connection"""
        setattr(self.websocket, '_peer_connected', False)
        self.connected = False