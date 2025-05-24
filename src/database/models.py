# models/models.py
from sqlalchemy import Column, String, JSON, ForeignKey, Text, DateTime, Float, Boolean, Integer, Enum, Table, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from datetime import datetime
import enum
import uuid


Base = declarative_base()

class AgentType(str, enum.Enum):
    base = "base"
    sales = "sales"
    support = "support"
    technical = "technical"
    custom = "custom"

class DocumentType(str, enum.Enum):
    faq = "faq"
    product = "product"
    policy = "policy"
    technical = "technical"
    custom = "custom"
    image = "image"  # New type for images


class DatabaseIntegrationType(str, enum.Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MSSQL = "mssql"
    ORACLE = "oracle"
    SQLITE = "sqlite"

class Company(Base):
    __tablename__ = 'Company'
    
    # Primary fields
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    name = Column(String(255), nullable=False)
    api_key = Column(String(255), unique=True, nullable=False)
    
    # Contact & Business Info
    phone_number = Column(String(20), unique=True, nullable=True)
    email = Column(String(255), unique=True, nullable=True)
    business_type = Column(String(100), nullable=True)
    website = Column(String(255), nullable=True)
    
    # Configuration
    settings = Column(JSONB, default=dict, nullable=False)
    prompt_templates = Column(JSONB, default=dict, nullable=False)
    active = Column(Boolean, default=True)
    
    # Qdrant Configuration
    qdrant_collection_name = Column(String(255), unique=True)
    vector_dimension = Column(Integer, default=1536)  # Default for text-embedding-3-small
    
    # Database Integration
    database_integrations = relationship("DatabaseIntegration", back_populates="company", cascade="all, delete-orphan")
    
    # Analytics & Metrics
    total_conversations = Column(Integer, default=0)
    average_response_time = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    agents = relationship("Agent", back_populates="company", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="company", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="company", cascade="all, delete-orphan")
    calls = relationship("Call", back_populates="company", cascade="all, delete-orphan")
    
    # Add new fields for image handling
    image_storage_limit = Column(Integer, default=10737418240)  # 10GB in bytes
    current_image_storage = Column(Integer, default=0)
    image_config = Column(JSONB, default={
        'enable_auto_tagging': True,
        'enable_explicit_content_detection': True,
        'retention_period_days': 365,
        'max_image_size': 10485760,  # 10MB
        'supported_formats': ['image/jpeg', 'image/png', 'image/gif']
    })



class DatabaseIntegration(Base):
    __tablename__ = 'DatabaseIntegration'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(String, ForeignKey('Company.id'))
    name = Column(String(255), nullable=False)
    type = Column(Enum(DatabaseIntegrationType), nullable=False)
    
    # Connection Details (encrypted)
    connection_details = Column(JSONB, nullable=False)
    
    # Schema Configuration
    schema_mapping = Column(JSONB, default=dict)  # Maps tables/columns to semantic meanings
    included_tables = Column(ARRAY(String), default=[])
    excluded_tables = Column(ARRAY(String), default=[])
    
    # Sync Configuration
    sync_frequency = Column(String(50), default='daily')  # real-time, hourly, daily, weekly
    last_sync = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    company = relationship("Company", back_populates="database_integrations")

class Document(Base):
    __tablename__ = 'Document'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(String, ForeignKey('Company.id'))
    agent_id = Column(String, ForeignKey('Agent.id'))
    
    name = Column(String(255), nullable=False)
    type = Column(Enum(DocumentType), nullable=False)
    content = Column(Text, nullable=False)
    
    # File Metadata
    file_type = Column(String(50))  # pdf, docx, txt, etc.
    file_size = Column(Integer)  # in bytes
    original_filename = Column(String(255))
    # Update file metadata for better image support
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    image_format = Column(String(20), nullable=True)
    
    
    # Add new fields for image handling
    is_image = Column(Boolean, default=False)
    image_content = Column(LargeBinary, nullable=True)  # For storing image data
    image_metadata = Column(JSONB, nullable=True)  # For storing image-specific metadata
    user_description = Column(Text, nullable=True)  # User-provided image description
    auto_description = Column(Text, nullable=True)  # AI-generated image description
    embedding = Column(ARRAY(Float), nullable=True)  # Vector embedding for the image

    
    # Vector Embedding
    embedding_id = Column(String(255))  # ID in Qdrant
    last_embedded = Column(DateTime)
    chunk_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    company = relationship("Company", back_populates="documents")
    agent = relationship("Agent", back_populates="documents")

class Agent(Base):
    __tablename__ = 'Agent'
    
    # Primary fields
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    name = Column(String(255), nullable=False)
    type = Column(Enum(AgentType), nullable=False)
    company_id = Column(String, ForeignKey('Company.id'))
    
    files = Column(ARRAY(String), default=[])

    
    # Core Configuration
    prompt = Column(Text, nullable=False)
    template_id = Column(String(255), nullable=True)
    additional_context = Column(JSONB, nullable=True)
    advanced_settings = Column(JSONB, nullable=True)
    
    # Vector Store Configuration
    knowledge_base_ids = Column(ARRAY(String), default=[])  # Document IDs for knowledge base
    database_integration_ids = Column(ARRAY(String), default=[])  # Database integration IDs
    
    # Query Configuration
    search_config = Column(JSONB, default={
        'score_threshold': 0.7,
        'limit': 5,
        'include_metadata': True
    })
    
    # Agent Settings
    confidence_threshold = Column(Float, default=0.7)
    max_response_tokens = Column(Integer, default=200)
    temperature = Column(Float, default=0.7)
    is_active = Column(Boolean, default=True)
    
    # Analytics & Performance
    total_interactions = Column(Integer, default=0)
    average_confidence = Column(Float, default=0.0)
    success_rate = Column(Float, default=0.0)
    average_response_time = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    company = relationship("Company", back_populates="agents")
    documents = relationship("Document", back_populates="agent")
    conversations = relationship("Conversation", back_populates="current_agent")
    interactions = relationship(
        "AgentInteraction",
        back_populates="agent",
        foreign_keys="[AgentInteraction.agent_id]",
        cascade="all, delete-orphan"
    )

    # Add new fields for image handling
    image_processing_enabled = Column(Boolean, default=False)
    image_processing_config = Column(JSONB, default={
        'max_images': 1000,
        'confidence_threshold': 0.7,
        'enable_auto_description': True
    })

# Add new ImageProcessingJob model for background processing
class ImageProcessingJob(Base):
    __tablename__ = 'ImageProcessingJob'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey('Document.id'))
    company_id = Column(String, ForeignKey('Company.id'))
    agent_id = Column(String, ForeignKey('Agent.id'))
    
    status = Column(String(50), default='pending')  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    
    processing_config = Column(JSONB, nullable=True)
    results = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    document = relationship("Document", backref="processing_jobs")
    company = relationship("Company")
    agent = relationship("Agent")


class Conversation(Base):
    __tablename__ = 'Conversation'
    
    # Primary fields
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String(255), nullable=False)
    company_id = Column(String, ForeignKey('Company.id'))
    current_agent_id = Column(String, ForeignKey('Agent.id'))
    
    # Conversation Data
    history = Column(JSONB, default=list, nullable=False)
    meta_data = Column(JSONB, nullable=True)
    
    # Analytics
    duration = Column(Float, default=0.0)  # in seconds
    messages_count = Column(Integer, default=0)
    sentiment_score = Column(Float, nullable=True)
    
    # Status
    status = Column(String(50), default="active")
    ended_by = Column(String(50), nullable=True)  # user/system/timeout
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    ended_at = Column(DateTime, nullable=True)
    
    # Relationships
    company = relationship("Company", back_populates="conversations")
    current_agent = relationship("Agent", back_populates="conversations")
    interactions = relationship("AgentInteraction", back_populates="conversation", cascade="all, delete-orphan")

class AgentInteraction(Base):
    __tablename__ = 'AgentInteraction'
    
    # Primary fields
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey('Agent.id'))
    conversation_id = Column(String, ForeignKey('Conversation.id'))
    
    # Interaction Details
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Performance Metrics
    response_time = Column(Float, nullable=False)  # in seconds
    tokens_used = Column(Integer, nullable=True)
    was_successful = Column(Boolean, nullable=True)
    
    # Context
    previous_agent_id = Column(String, ForeignKey('Agent.id'), nullable=True)
    context_window = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    agent = relationship("Agent", back_populates="interactions", foreign_keys=[agent_id])
    conversation = relationship("Conversation", back_populates="interactions")
    previous_agent = relationship(
        "Agent", 
        foreign_keys=[previous_agent_id],
        overlaps="previous_interactions",  # Added this line
        viewonly=True  # Added this line
    )
    
    
class Call(Base):
    __tablename__ = 'Call'
    
    # Primary fields
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(String, ForeignKey('Company.id'))
    conversation_id = Column(String, ForeignKey('Conversation.id'), nullable=True)
    
    # Call Details
    call_sid = Column(String(255), unique=True)
    from_number = Column(String(20), nullable=False)
    to_number = Column(String(20), nullable=False)
    
    # Status and Duration
    status = Column(String(50), nullable=False)
    duration = Column(Float)  # in seconds
    
    # Media
    recording_url = Column(String(255))
    transcription = Column(Text)
    
    # Analytics
    cost = Column(Float, default=0.0)
    quality_score = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    answered_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    
    # Relationships
    company = relationship("Company", back_populates="calls")