"""
Agent domain entities for the AI voice calling system.

This module contains entities related to AI agents, their configurations,
capabilities, and performance metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List, Set, Union
from uuid import uuid4


class AgentType(Enum):
    """Types of AI agents."""
    CONVERSATIONAL = "conversational"
    FUNCTION_CALLING = "function_calling"
    VOICE_SPECIALIZED = "voice_specialized"
    MULTIMODAL = "multimodal"
    CUSTOMER_SERVICE = "customer_service"
    SALES = "sales"
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL_ASSISTANT = "general_assistant"


class AgentCapability(Enum):
    """Capabilities that an agent can have."""
    TEXT_GENERATION = "text_generation"
    VOICE_SYNTHESIS = "voice_synthesis"
    VOICE_RECOGNITION = "voice_recognition"
    FUNCTION_CALLING = "function_calling"
    CONTEXT_MEMORY = "context_memory"
    MULTIMODAL_INPUT = "multimodal_input"
    REAL_TIME_PROCESSING = "real_time_processing"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    LANGUAGE_TRANSLATION = "language_translation"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    CALENDAR_INTEGRATION = "calendar_integration"
    EMAIL_INTEGRATION = "email_integration"
    WEB_SEARCH = "web_search"
    DOCUMENT_ANALYSIS = "document_analysis"
    IMAGE_PROCESSING = "image_processing"


@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for an agent."""
    
    # Core model configuration
    model_name: str
    model_version: str = "latest"
    provider: str = "openai"  # openai, anthropic, etc.
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Voice configuration
    voice_model: Optional[str] = None
    voice_speed: float = 1.0
    voice_pitch: float = 1.0
    preferred_language: str = "en"
    
    # Context and memory
    context_window_size: int = 4000
    memory_enabled: bool = True
    max_conversation_history: int = 100
    
    # Function calling
    function_calling_enabled: bool = False
    available_functions: List[str] = field(default_factory=list)
    function_timeout: int = 30
    
    # Response configuration
    response_format: str = "text"  # text, json, structured
    streaming_enabled: bool = True
    max_response_time: float = 30.0
    
    # Safety and filtering
    content_filter_enabled: bool = True
    safety_level: str = "medium"  # low, medium, high
    allowed_topics: List[str] = field(default_factory=list)
    blocked_topics: List[str] = field(default_factory=list)
    
    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 10000
    concurrent_requests: int = 5
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        
        if self.context_window_size <= 0:
            raise ValueError("context_window_size must be positive")
    
    def with_temperature(self, temperature: float) -> 'AgentConfig':
        """Create new config with different temperature."""
        return AgentConfig(
            **{**self.__dict__, 'temperature': temperature}
        )
    
    def with_max_tokens(self, max_tokens: int) -> 'AgentConfig':
        """Create new config with different max_tokens."""
        return AgentConfig(
            **{**self.__dict__, 'max_tokens': max_tokens}
        )
    
    def with_functions(self, functions: List[str]) -> 'AgentConfig':
        """Create new config with function calling enabled."""
        return AgentConfig(
            **{
                **self.__dict__,
                'function_calling_enabled': True,
                'available_functions': functions
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "voice_model": self.voice_model,
            "voice_speed": self.voice_speed,
            "voice_pitch": self.voice_pitch,
            "preferred_language": self.preferred_language,
            "context_window_size": self.context_window_size,
            "memory_enabled": self.memory_enabled,
            "max_conversation_history": self.max_conversation_history,
            "function_calling_enabled": self.function_calling_enabled,
            "available_functions": self.available_functions,
            "function_timeout": self.function_timeout,
            "response_format": self.response_format,
            "streaming_enabled": self.streaming_enabled,
            "max_response_time": self.max_response_time,
            "content_filter_enabled": self.content_filter_enabled,
            "safety_level": self.safety_level,
            "allowed_topics": self.allowed_topics,
            "blocked_topics": self.blocked_topics,
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
            "concurrent_requests": self.concurrent_requests
        }


@dataclass(frozen=True)
class AgentMetrics:
    """Immutable performance metrics for an agent."""
    
    # Usage statistics
    total_conversations: int = 0
    total_messages: int = 0
    total_tokens_processed: int = 0
    total_tokens_generated: int = 0
    
    # Performance metrics
    average_response_time: float = 0.0
    average_tokens_per_response: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    
    # Voice metrics
    total_voice_minutes: float = 0.0
    average_voice_quality_score: float = 0.0
    voice_synthesis_count: int = 0
    
    # Function calling metrics
    function_calls_made: int = 0
    function_calls_successful: int = 0
    average_function_execution_time: float = 0.0
    
    # User satisfaction
    average_satisfaction_score: Optional[float] = None
    feedback_count: int = 0
    
    # Operational metrics
    uptime_percentage: float = 100.0
    last_active_time: Optional[datetime] = None
    
    def add_conversation(self) -> 'AgentMetrics':
        """Create new metrics with incremented conversation count."""
        return AgentMetrics(
            total_conversations=self.total_conversations + 1,
            total_messages=self.total_messages,
            total_tokens_processed=self.total_tokens_processed,
            total_tokens_generated=self.total_tokens_generated,
            average_response_time=self.average_response_time,
            average_tokens_per_response=self.average_tokens_per_response,
            success_rate=self.success_rate,
            error_count=self.error_count,
            total_voice_minutes=self.total_voice_minutes,
            average_voice_quality_score=self.average_voice_quality_score,
            voice_synthesis_count=self.voice_synthesis_count,
            function_calls_made=self.function_calls_made,
            function_calls_successful=self.function_calls_successful,
            average_function_execution_time=self.average_function_execution_time,
            average_satisfaction_score=self.average_satisfaction_score,
            feedback_count=self.feedback_count,
            uptime_percentage=self.uptime_percentage,
            last_active_time=datetime.now(timezone.utc)
        )
    
    def add_response(
        self,
        response_time: float,
        tokens_processed: int,
        tokens_generated: int,
        success: bool = True
    ) -> 'AgentMetrics':
        """Create new metrics with response data."""
        new_message_count = self.total_messages + 1
        new_error_count = self.error_count + (0 if success else 1)
        
        # Calculate new averages
        new_avg_response_time = (
            (self.average_response_time * self.total_messages + response_time) / new_message_count
        )
        new_avg_tokens = (
            (self.average_tokens_per_response * self.total_messages + tokens_generated) / new_message_count
        )
        new_success_rate = (new_message_count - new_error_count) / new_message_count
        
        return AgentMetrics(
            total_conversations=self.total_conversations,
            total_messages=new_message_count,
            total_tokens_processed=self.total_tokens_processed + tokens_processed,
            total_tokens_generated=self.total_tokens_generated + tokens_generated,
            average_response_time=new_avg_response_time,
            average_tokens_per_response=new_avg_tokens,
            success_rate=new_success_rate,
            error_count=new_error_count,
            total_voice_minutes=self.total_voice_minutes,
            average_voice_quality_score=self.average_voice_quality_score,
            voice_synthesis_count=self.voice_synthesis_count,
            function_calls_made=self.function_calls_made,
            function_calls_successful=self.function_calls_successful,
            average_function_execution_time=self.average_function_execution_time,
            average_satisfaction_score=self.average_satisfaction_score,
            feedback_count=self.feedback_count,
            uptime_percentage=self.uptime_percentage,
            last_active_time=datetime.now(timezone.utc)
        )
    
    def add_voice_session(self, duration_minutes: float, quality_score: float) -> 'AgentMetrics':
        """Create new metrics with voice session data."""
        new_voice_count = self.voice_synthesis_count + 1
        new_avg_quality = (
            (self.average_voice_quality_score * self.voice_synthesis_count + quality_score) / new_voice_count
        )
        
        return AgentMetrics(
            total_conversations=self.total_conversations,
            total_messages=self.total_messages,
            total_tokens_processed=self.total_tokens_processed,
            total_tokens_generated=self.total_tokens_generated,
            average_response_time=self.average_response_time,
            average_tokens_per_response=self.average_tokens_per_response,
            success_rate=self.success_rate,
            error_count=self.error_count,
            total_voice_minutes=self.total_voice_minutes + duration_minutes,
            average_voice_quality_score=new_avg_quality,
            voice_synthesis_count=new_voice_count,
            function_calls_made=self.function_calls_made,
            function_calls_successful=self.function_calls_successful,
            average_function_execution_time=self.average_function_execution_time,
            average_satisfaction_score=self.average_satisfaction_score,
            feedback_count=self.feedback_count,
            uptime_percentage=self.uptime_percentage,
            last_active_time=datetime.now(timezone.utc)
        )
    
    def add_function_call(self, execution_time: float, success: bool = True) -> 'AgentMetrics':
        """Create new metrics with function call data."""
        new_calls_made = self.function_calls_made + 1
        new_calls_successful = self.function_calls_successful + (1 if success else 0)
        
        new_avg_execution_time = (
            (self.average_function_execution_time * self.function_calls_made + execution_time) / new_calls_made
        )
        
        return AgentMetrics(
            total_conversations=self.total_conversations,
            total_messages=self.total_messages,
            total_tokens_processed=self.total_tokens_processed,
            total_tokens_generated=self.total_tokens_generated,
            average_response_time=self.average_response_time,
            average_tokens_per_response=self.average_tokens_per_response,
            success_rate=self.success_rate,
            error_count=self.error_count,
            total_voice_minutes=self.total_voice_minutes,
            average_voice_quality_score=self.average_voice_quality_score,
            voice_synthesis_count=self.voice_synthesis_count,
            function_calls_made=new_calls_made,
            function_calls_successful=new_calls_successful,
            average_function_execution_time=new_avg_execution_time,
            average_satisfaction_score=self.average_satisfaction_score,
            feedback_count=self.feedback_count,
            uptime_percentage=self.uptime_percentage,
            last_active_time=datetime.now(timezone.utc)
        )
    
    def add_feedback(self, satisfaction_score: float) -> 'AgentMetrics':
        """Create new metrics with user feedback."""
        new_feedback_count = self.feedback_count + 1
        
        if self.average_satisfaction_score is None:
            new_avg_satisfaction = satisfaction_score
        else:
            new_avg_satisfaction = (
                (self.average_satisfaction_score * self.feedback_count + satisfaction_score) / new_feedback_count
            )
        
        return AgentMetrics(
            total_conversations=self.total_conversations,
            total_messages=self.total_messages,
            total_tokens_processed=self.total_tokens_processed,
            total_tokens_generated=self.total_tokens_generated,
            average_response_time=self.average_response_time,
            average_tokens_per_response=self.average_tokens_per_response,
            success_rate=self.success_rate,
            error_count=self.error_count,
            total_voice_minutes=self.total_voice_minutes,
            average_voice_quality_score=self.average_voice_quality_score,
            voice_synthesis_count=self.voice_synthesis_count,
            function_calls_made=self.function_calls_made,
            function_calls_successful=self.function_calls_successful,
            average_function_execution_time=self.average_function_execution_time,
            average_satisfaction_score=new_avg_satisfaction,
            feedback_count=new_feedback_count,
            uptime_percentage=self.uptime_percentage,
            last_active_time=datetime.now(timezone.utc)
        )
    
    @property
    def function_success_rate(self) -> float:
        """Calculate function call success rate."""
        if self.function_calls_made == 0:
            return 1.0
        return self.function_calls_successful / self.function_calls_made
    
    @property
    def average_conversation_length(self) -> float:
        """Calculate average messages per conversation."""
        if self.total_conversations == 0:
            return 0.0
        return self.total_messages / self.total_conversations


@dataclass
class Agent:
    """
    Represents an AI agent with comprehensive configuration and state management.
    
    This entity encapsulates all the business logic related to an AI agent,
    including its capabilities, configuration, performance metrics, and state.
    """
    
    # Core identifiers
    id: str
    name: str
    agent_type: AgentType
    
    # Agent metadata
    description: Optional[str] = None
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Capabilities and configuration
    capabilities: Optional[Set[AgentCapability]] = None
    config: AgentConfig = field(default_factory=lambda: AgentConfig(model_name="gpt-3.5-turbo"))
    
    # State and status
    is_active: bool = True
    is_available: bool = True
    current_load: int = 0  # Number of active conversations
    max_concurrent_conversations: int = 10
    
    # Performance and metrics
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    
    # Personalization and learning
    system_prompt: str = "You are a helpful AI assistant."
    personality_traits: Dict[str, Any] = field(default_factory=dict)
    knowledge_base: Optional[str] = None
    
    # Operational settings
    deployment_environment: str = "production"  # development, staging, production
    health_check_interval: int = 60  # seconds
    auto_scaling_enabled: bool = False
    
    # Integration settings
    webhook_url: Optional[str] = None
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    # In the Agent class
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.id:
            raise ValueError("id cannot be empty")
        if not self.name:
            raise ValueError("name cannot be empty")
        
        # Ensure timestamps are timezone-aware
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if self.updated_at.tzinfo is None:
            self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)
            
        # ** THE FIX IS HERE **
        # Only set default capabilities if none were provided at all.
        if self.capabilities is None:
            self._set_default_capabilities()
        
        # If an empty set was passed, self.capabilities will be set() and this block is skipped.

    def _set_default_capabilities(self) -> None:
        """Set default capabilities based on agent type."""
        # This method is now only called when self.capabilities is None.
        
        if self.agent_type == AgentType.CONVERSATIONAL:
            self.capabilities = {
                AgentCapability.TEXT_GENERATION,
                AgentCapability.CONTEXT_MEMORY,
                AgentCapability.SENTIMENT_ANALYSIS
            }
        elif self.agent_type == AgentType.VOICE_SPECIALIZED:
            self.capabilities = {
                AgentCapability.TEXT_GENERATION,
                AgentCapability.VOICE_SYNTHESIS,
                AgentCapability.VOICE_RECOGNITION,
                AgentCapability.REAL_TIME_PROCESSING
            }
        elif self.agent_type == AgentType.FUNCTION_CALLING:
            self.capabilities = {
                AgentCapability.TEXT_GENERATION,
                AgentCapability.FUNCTION_CALLING,
                AgentCapability.CONTEXT_MEMORY
            }
        elif self.agent_type == AgentType.CUSTOMER_SERVICE:
            self.capabilities = {
                AgentCapability.TEXT_GENERATION,
                AgentCapability.SENTIMENT_ANALYSIS,
                AgentCapability.KNOWLEDGE_RETRIEVAL,
                AgentCapability.FUNCTION_CALLING
            }
        else:
            # Default for any other type
            self.capabilities = {AgentCapability.TEXT_GENERATION}
    
    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to the agent."""
        self.capabilities.add(capability)
        self.updated_at = datetime.now(timezone.utc)
    
    def remove_capability(self, capability: AgentCapability) -> None:
        """Remove a capability from the agent."""
        self.capabilities.discard(capability)
        self.updated_at = datetime.now(timezone.utc)
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities
    
    def update_config(self, **config_updates) -> None:
        """Update agent configuration."""
        current_config_dict = self.config.to_dict()
        current_config_dict.update(config_updates)
        self.config = AgentConfig(**current_config_dict)
        self.updated_at = datetime.now(timezone.utc)
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for the agent."""
        self.system_prompt = prompt
        self.updated_at = datetime.now(timezone.utc)
    
    def add_personality_trait(self, trait: str, value: Any) -> None:
        """Add or update a personality trait."""
        self.personality_traits[trait] = value
        self.updated_at = datetime.now(timezone.utc)
    
    def remove_personality_trait(self, trait: str) -> None:
        """Remove a personality trait."""
        self.personality_traits.pop(trait, None)
        self.updated_at = datetime.now(timezone.utc)
    
    def start_conversation(self) -> bool:
        """Start a new conversation if capacity allows."""
        if not self.is_available or not self.is_active:
            return False
        
        if self.current_load >= self.max_concurrent_conversations:
            return False
        
        self.current_load += 1
        self.metrics = self.metrics.add_conversation()
        self.updated_at = datetime.now(timezone.utc)
        return True
    
    def end_conversation(self) -> None:
        """End a conversation and decrease load."""
        if self.current_load > 0:
            self.current_load -= 1
        self.updated_at = datetime.now(timezone.utc)
    
    def record_response(
        self,
        response_time: float,
        tokens_processed: int,
        tokens_generated: int,
        success: bool = True
    ) -> None:
        """Record a response and update metrics."""
        self.metrics = self.metrics.add_response(
            response_time=response_time,
            tokens_processed=tokens_processed,
            tokens_generated=tokens_generated,
            success=success
        )
        self.updated_at = datetime.now(timezone.utc)
    
    def record_voice_session(self, duration_minutes: float, quality_score: float) -> None:
        """Record a voice session and update metrics."""
        self.metrics = self.metrics.add_voice_session(duration_minutes, quality_score)
        self.updated_at = datetime.now(timezone.utc)
    
    def record_function_call(self, execution_time: float, success: bool = True) -> None:
        """Record a function call and update metrics."""
        self.metrics = self.metrics.add_function_call(execution_time, success)
        self.updated_at = datetime.now(timezone.utc)
    
    def record_feedback(self, satisfaction_score: float) -> None:
        """Record user feedback and update metrics."""
        if not 0.0 <= satisfaction_score <= 5.0:
            raise ValueError("satisfaction_score must be between 0.0 and 5.0")
        
        self.metrics = self.metrics.add_feedback(satisfaction_score)
        self.updated_at = datetime.now(timezone.utc)
    
    def activate(self) -> None:
        """Activate the agent."""
        self.is_active = True
        self.updated_at = datetime.now(timezone.utc)
    
    def deactivate(self) -> None:
        """Deactivate the agent."""
        self.is_active = False
        self.updated_at = datetime.now(timezone.utc)
    
    def make_available(self) -> None:
        """Make the agent available for new conversations."""
        self.is_available = True
        self.updated_at = datetime.now(timezone.utc)
    
    def make_unavailable(self) -> None:
        """Make the agent unavailable for new conversations."""
        self.is_available = False
        self.updated_at = datetime.now(timezone.utc)
    
    def is_overloaded(self) -> bool:
        """Check if agent is overloaded."""
        return self.current_load >= self.max_concurrent_conversations
    
    def get_load_percentage(self) -> float:
        """Get current load as percentage of maximum capacity."""
        if self.max_concurrent_conversations == 0:
            return 0.0
        return (self.current_load / self.max_concurrent_conversations) * 100
    
    def can_handle_new_conversation(self) -> bool:
        """Check if agent can handle a new conversation."""
        return (
            self.is_active and 
            self.is_available and 
            not self.is_overloaded()
        )
    
    def get_uptime(self) -> Optional[float]:
        """Get agent uptime in seconds since creation."""
        if not self.is_active:
            return None
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()
    
    def get_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        if self.metrics.total_messages == 0:
            return 100.0
        
        # Weighted performance calculation
        success_weight = 0.4
        response_time_weight = 0.3
        satisfaction_weight = 0.3
        
        # Success rate component (0-100)
        success_component = self.metrics.success_rate * 100
        
        # Response time component (inverse relationship, faster = better)
        # Assume 5 seconds is perfect (100), 30 seconds is poor (0)
        max_acceptable_time = 30.0
        response_time_component = max(
            0, 
            (max_acceptable_time - self.metrics.average_response_time) / max_acceptable_time * 100
        )
        
        # Satisfaction component
        satisfaction_component = 100.0
        if self.metrics.average_satisfaction_score is not None:
            satisfaction_component = (self.metrics.average_satisfaction_score / 5.0) * 100
        
        return (
            success_component * success_weight +
            response_time_component * response_time_weight +
            satisfaction_component * satisfaction_weight
        )
    
    def validate_config(self) -> List[str]:
        """Validate agent configuration and return list of issues."""
        issues = []
        
        # Check required capabilities for agent type
        if self.agent_type == AgentType.VOICE_SPECIALIZED:
            required_caps = {AgentCapability.VOICE_SYNTHESIS, AgentCapability.VOICE_RECOGNITION}
            missing_caps = required_caps - self.capabilities
            if missing_caps:
                missing_names = [cap.value for cap in missing_caps]
                issues.append(f"Voice specialized agent missing capabilities: {missing_names}")
        
        if self.agent_type == AgentType.FUNCTION_CALLING:
            if not self.has_capability(AgentCapability.FUNCTION_CALLING):
                issues.append("Function calling agent missing FUNCTION_CALLING capability")
            if not self.config.function_calling_enabled:
                issues.append("Function calling agent has function calling disabled in config")
        
        # Check configuration consistency
        if self.has_capability(AgentCapability.VOICE_SYNTHESIS) and not self.config.voice_model:
            issues.append("Voice synthesis capability enabled but no voice model configured")
        
        if self.config.function_calling_enabled and not self.config.available_functions:
            issues.append("Function calling enabled but no functions configured")
        
        # Check resource limits
        if self.max_concurrent_conversations <= 0:
            issues.append("max_concurrent_conversations must be positive")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "agent_type": self.agent_type.value,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "capabilities": [cap.value for cap in self.capabilities],
            "config": self.config.to_dict(),
            "is_active": self.is_active,
            "is_available": self.is_available,
            "current_load": self.current_load,
            "max_concurrent_conversations": self.max_concurrent_conversations,
            "system_prompt": self.system_prompt,
            "personality_traits": self.personality_traits,
            "knowledge_base": self.knowledge_base,
            "deployment_environment": self.deployment_environment,
            "health_check_interval": self.health_check_interval,
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "webhook_url": self.webhook_url,
            "metrics": {
                "total_conversations": self.metrics.total_conversations,
                "total_messages": self.metrics.total_messages,
                "total_tokens_processed": self.metrics.total_tokens_processed,
                "total_tokens_generated": self.metrics.total_tokens_generated,
                "average_response_time": self.metrics.average_response_time,
                "average_tokens_per_response": self.metrics.average_tokens_per_response,
                "success_rate": self.metrics.success_rate,
                "error_count": self.metrics.error_count,
                "total_voice_minutes": self.metrics.total_voice_minutes,
                "average_voice_quality_score": self.metrics.average_voice_quality_score,
                "voice_synthesis_count": self.metrics.voice_synthesis_count,
                "function_calls_made": self.metrics.function_calls_made,
                "function_calls_successful": self.metrics.function_calls_successful,
                "average_function_execution_time": self.metrics.average_function_execution_time,
                "function_success_rate": self.metrics.function_success_rate,
                "average_satisfaction_score": self.metrics.average_satisfaction_score,
                "feedback_count": self.metrics.feedback_count,
                "uptime_percentage": self.metrics.uptime_percentage,
                "last_active_time": self.metrics.last_active_time.isoformat() if self.metrics.last_active_time else None,
                "average_conversation_length": self.metrics.average_conversation_length
            },
            "load_percentage": self.get_load_percentage(),
            "performance_score": self.get_performance_score(),
            "uptime_seconds": self.get_uptime(),
            "can_handle_new_conversation": self.can_handle_new_conversation()
        }
    
    def __str__(self) -> str:
        status = "active" if self.is_active else "inactive"
        availability = "available" if self.is_available else "unavailable"
        return f"Agent(id={self.id}, name={self.name}, type={self.agent_type.value}, {status}, {availability}, load={self.current_load}/{self.max_concurrent_conversations})"