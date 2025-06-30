# src/infrastructure/database/models.py
"""
SQLAlchemy database models for the AI voice calling system.
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, BigInteger, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()

class CompanyStatus(enum.Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"

class EventType(enum.Enum):
    CONNECTION = "connection"
    DISCONNECTION = "disconnection"
    MESSAGE = "message"
    VOICE_START = "voice_start"
    VOICE_END = "voice_end"
    ERROR = "error"

class MessageType(enum.Enum):
    USER_TEXT = "user_text"
    USER_AUDIO = "user_audio"
    AGENT_TEXT = "agent_text"
    AGENT_AUDIO = "agent_audio"
    SYSTEM = "system"
    ERROR = "error"

class Company(Base):
    __tablename__ = "companies"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    api_key = Column(String(255), unique=True, nullable=False, index=True)
    status = Column(Enum(CompanyStatus), default=CompanyStatus.ACTIVE)
    
    # Usage limits
    max_connections = Column(Integer, default=100)
    max_requests_per_minute = Column(Integer, default=60)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    session_events = relationship("SessionEvent", back_populates="company")
    conversations = relationship("ConversationRecord", back_populates="company")
    
    __table_args__ = (
        Index('idx_company_api_key', 'api_key'),
        Index('idx_company_status', 'status'),
    )

class SessionEvent(Base):
    __tablename__ = "session_events"
    
    id = Column(String(100), primary_key=True)
    company_id = Column(String(50), ForeignKey("companies.id"), nullable=False)
    client_id = Column(String(100), nullable=False)
    session_id = Column(String(100), nullable=True)
    
    event_type = Column(Enum(EventType), nullable=False)
    timestamp = Column(DateTime, default=func.now())
    
    # Session metrics
    session_duration = Column(Float, nullable=True)
    message_count = Column(Integer, default=0)
    token_count = Column(Integer, default=0)
    voice_duration = Column(Float, nullable=True)
    
    # Error information
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    
    # Additional metadata
    metadata = Column(JSON, default=dict)
    
    # Relationships
    company = relationship("Company", back_populates="session_events")
    
    __table_args__ = (
        Index('idx_session_company_id', 'company_id'),
        Index('idx_session_client_id', 'client_id'),
        Index('idx_session_event_type', 'event_type'),
        Index('idx_session_timestamp', 'timestamp'),
    )

class ConversationRecord(Base):
    __tablename__ = "conversations"
    
    id = Column(String(100), primary_key=True)
    client_id = Column(String(100), nullable=False)
    company_id = Column(String(50), ForeignKey("companies.id"), nullable=False)
    agent_id = Column(String(100), nullable=True)
    
    # Conversation metadata
    title = Column(String(500), nullable=True)
    state = Column(String(50), default="initialized")
    context = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    ended_at = Column(DateTime, nullable=True)
    
    # Metrics
    message_count = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    voice_duration = Column(Float, default=0.0)
    
    # Relationships
    company = relationship("Company", back_populates="conversations")
    messages = relationship("MessageRecord", back_populates="conversation", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_conversation_client_id', 'client_id'),
        Index('idx_conversation_company_id', 'company_id'),
        Index('idx_conversation_agent_id', 'agent_id'),
        Index('idx_conversation_created_at', 'created_at'),
    )

class MessageRecord(Base):
    __tablename__ = "messages"
    
    id = Column(String(100), primary_key=True)
    conversation_id = Column(String(100), ForeignKey("conversations.id"), nullable=False)
    
    # Message content
    message_type = Column(Enum(MessageType), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    
    # Message metadata
    sender_id = Column(String(100), nullable=True)
    agent_id = Column(String(100), nullable=True)
    client_id = Column(String(100), nullable=True)
    
    # Processing metrics
    tokens_used = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)
    confidence_score = Column(Float, nullable=True)
    
    # Audio-specific fields
    audio_duration = Column(Float, nullable=True)
    audio_format = Column(String(50), nullable=True)
    transcription_confidence = Column(Float, nullable=True)
    
    # Function call fields
    function_name = Column(String(255), nullable=True)
    function_args = Column(JSON, nullable=True)
    function_result = Column(JSON, nullable=True)
    
    # Error fields
    error_code = Column(String(100), nullable=True)
    error_details = Column(JSON, nullable=True)
    
    # Additional metadata
    metadata = Column(JSON, default=dict)
    
    # Relationships
    conversation = relationship("ConversationRecord", back_populates="messages")
    
    __table_args__ = (
        Index('idx_message_conversation_id', 'conversation_id'),
        Index('idx_message_timestamp', 'timestamp'),
        Index('idx_message_type', 'message_type'),
        Index('idx_message_client_id', 'client_id'),
    )

class AgentRecord(Base):
    __tablename__ = "agents"
    
    id = Column(String(100), primary_key=True)
    name = Column(String(255), nullable=False)
    agent_type = Column(String(100), nullable=False)
    
    # Agent configuration
    config = Column(JSON, nullable=False)
    capabilities = Column(JSON, default=list)
    system_prompt = Column(Text, nullable=True)
    
    # State
    is_active = Column(Boolean, default=True)
    is_available = Column(Boolean, default=True)
    current_load = Column(Integer, default=0)
    max_concurrent = Column(Integer, default=10)
    
    # Metrics
    total_conversations = Column(BigInteger, default=0)
    total_messages = Column(BigInteger, default=0)
    total_tokens = Column(BigInteger, default=0)
    average_response_time = Column(Float, default=0.0)
    success_rate = Column(Float, default=1.0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_active = Column(DateTime, nullable=True)
    
    __table_args__ = (
        Index('idx_agent_type', 'agent_type'),
        Index('idx_agent_active', 'is_active'),
        Index('idx_agent_available', 'is_available'),
    )