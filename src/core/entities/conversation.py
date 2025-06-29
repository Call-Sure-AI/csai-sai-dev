from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

class ConversationStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"

@dataclass
class Message:
    """Individual message in conversation"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: int = 0

@dataclass
class Conversation:
    """Conversation domain entity"""
    id: str
    customer_id: str
    company_id: str
    agent_id: Optional[str] = None
    
    # State
    status: ConversationStatus = ConversationStatus.ACTIVE
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    total_tokens: int = 0
    message_count: int = 0
    
    def add_message(self, role: str, content: str, tokens: int = 0, metadata: Dict[str, Any] = None) -> None:
        """Add message to conversation"""
        message = Message(
            role=role,
            content=content,
            tokens=tokens,
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        self.message_count += 1
        self.total_tokens += tokens
        self.updated_at = datetime.utcnow()
    
    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """Get recent messages for context"""
        return self.messages[-limit:] if self.messages else []
    
    def get_context_for_ai(self, max_messages: int = 5) -> List[Dict[str, str]]:
        """Get conversation context formatted for AI"""
        recent = self.get_recent_messages(max_messages)
        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent
        ]
    
    def calculate_duration(self) -> float:
        """Calculate conversation duration in seconds"""
        return (self.updated_at - self.created_at).total_seconds()