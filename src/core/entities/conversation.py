"""
Conversation domain entities for the AI voice calling system.

This module contains entities related to conversations, messages,
and their associated state and metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from uuid import uuid4


class MessageType(Enum):
    """Types of messages in a conversation."""
    USER_TEXT = "user_text"
    USER_AUDIO = "user_audio"
    AGENT_TEXT = "agent_text"
    AGENT_AUDIO = "agent_audio"
    SYSTEM = "system"
    ERROR = "error"
    FUNCTION_CALL = "function_call"
    FUNCTION_RESPONSE = "function_response"


class ConversationState(Enum):
    """Current state of a conversation."""
    INITIALIZED = "initialized"
    ACTIVE = "active"
    WAITING_FOR_INPUT = "waiting_for_input"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass(frozen=True)
class Message:
    """
    Immutable message entity representing a single message in a conversation.
    """
    id: str
    conversation_id: str
    message_type: MessageType
    content: str
    timestamp: datetime
    
    # Optional metadata
    sender_id: Optional[str] = None
    agent_id: Optional[str] = None
    client_id: Optional[str] = None
    
    # Message-specific data
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    processing_time: float = 0.0
    confidence_score: Optional[float] = None
    
    # Audio-specific fields
    audio_duration: Optional[float] = None
    audio_format: Optional[str] = None
    transcription_confidence: Optional[float] = None
    
    # Function call fields
    function_name: Optional[str] = None
    function_args: Optional[Dict[str, Any]] = None
    function_result: Optional[Any] = None
    
    # Error fields
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.id:
            object.__setattr__(self, 'id', str(uuid4()))
        
        if not self.conversation_id:
            raise ValueError("conversation_id cannot be empty")
        
        if not self.content and self.message_type not in [MessageType.FUNCTION_CALL, MessageType.SYSTEM]:
            raise ValueError("content cannot be empty for this message type")
        
        # Ensure timestamp is timezone-aware
        if self.timestamp.tzinfo is None:
            object.__setattr__(self, 'timestamp', self.timestamp.replace(tzinfo=timezone.utc))
    
    @classmethod
    def create_user_text(
        cls,
        conversation_id: str,
        content: str,
        client_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """Create a user text message."""
        return cls(
            id=str(uuid4()),
            conversation_id=conversation_id,
            message_type=MessageType.USER_TEXT,
            content=content,
            timestamp=datetime.now(timezone.utc),
            client_id=client_id,
            metadata=metadata or {}
        )
    
    @classmethod
    def create_user_audio(
        cls,
        conversation_id: str,
        content: str,  # Transcribed text
        client_id: str,
        audio_duration: float,
        audio_format: str = "webm",
        transcription_confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """Create a user audio message."""
        return cls(
            id=str(uuid4()),
            conversation_id=conversation_id,
            message_type=MessageType.USER_AUDIO,
            content=content,
            timestamp=datetime.now(timezone.utc),
            client_id=client_id,
            audio_duration=audio_duration,
            audio_format=audio_format,
            transcription_confidence=transcription_confidence,
            metadata=metadata or {}
        )
    
    @classmethod
    def create_agent_response(
        cls,
        conversation_id: str,
        content: str,
        agent_id: str,
        message_type: MessageType = MessageType.AGENT_TEXT,
        tokens_used: int = 0,
        processing_time: float = 0.0,
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """Create an agent response message."""
        return cls(
            id=str(uuid4()),
            conversation_id=conversation_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(timezone.utc),
            agent_id=agent_id,
            tokens_used=tokens_used,
            processing_time=processing_time,
            confidence_score=confidence_score,
            metadata=metadata or {}
        )
    
    @classmethod
    def create_function_call(
        cls,
        conversation_id: str,
        function_name: str,
        function_args: Dict[str, Any],
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """Create a function call message."""
        return cls(
            id=str(uuid4()),
            conversation_id=conversation_id,
            message_type=MessageType.FUNCTION_CALL,
            content=f"Function call: {function_name}",
            timestamp=datetime.now(timezone.utc),
            agent_id=agent_id,
            function_name=function_name,
            function_args=function_args,
            metadata=metadata or {}
        )
    
    @classmethod
    def create_function_response(
        cls,
        conversation_id: str,
        function_name: str,
        function_result: Any,
        processing_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """Create a function response message."""
        return cls(
            id=str(uuid4()),
            conversation_id=conversation_id,
            message_type=MessageType.FUNCTION_RESPONSE,
            content=f"Function response: {function_name}",
            timestamp=datetime.now(timezone.utc),
            function_name=function_name,
            function_result=function_result,
            processing_time=processing_time,
            metadata=metadata or {}
        )
    
    @classmethod
    def create_system_message(
        cls,
        conversation_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """Create a system message."""
        return cls(
            id=str(uuid4()),
            conversation_id=conversation_id,
            message_type=MessageType.SYSTEM,
            content=content,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
    
    @classmethod
    def create_error_message(
        cls,
        conversation_id: str,
        error_code: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """Create an error message."""
        return cls(
            id=str(uuid4()),
            conversation_id=conversation_id,
            message_type=MessageType.ERROR,
            content=error_message,
            timestamp=datetime.now(timezone.utc),
            error_code=error_code,
            error_details=error_details or {},
            metadata=metadata or {}
        )
    
    def is_user_message(self) -> bool:
        """Check if message is from user."""
        return self.message_type in [MessageType.USER_TEXT, MessageType.USER_AUDIO]
    
    def is_agent_message(self) -> bool:
        """Check if message is from agent."""
        return self.message_type in [MessageType.AGENT_TEXT, MessageType.AGENT_AUDIO]
    
    def is_function_related(self) -> bool:
        """Check if message is function-related."""
        return self.message_type in [MessageType.FUNCTION_CALL, MessageType.FUNCTION_RESPONSE]
    
    def has_audio_data(self) -> bool:
        """Check if message contains audio data."""
        return self.message_type in [MessageType.USER_AUDIO, MessageType.AGENT_AUDIO]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        result = {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tokens_used": self.tokens_used,
            "processing_time": self.processing_time
        }
        
        # Add optional fields if present
        if self.sender_id:
            result["sender_id"] = self.sender_id
        if self.agent_id:
            result["agent_id"] = self.agent_id
        if self.client_id:
            result["client_id"] = self.client_id
        if self.confidence_score is not None:
            result["confidence_score"] = self.confidence_score
        if self.audio_duration is not None:
            result["audio_duration"] = self.audio_duration
        if self.audio_format:
            result["audio_format"] = self.audio_format
        if self.transcription_confidence is not None:
            result["transcription_confidence"] = self.transcription_confidence
        if self.function_name:
            result["function_name"] = self.function_name
        if self.function_args:
            result["function_args"] = self.function_args
        if self.function_result is not None:
            result["function_result"] = self.function_result
        if self.error_code:
            result["error_code"] = self.error_code
        if self.error_details:
            result["error_details"] = self.error_details
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result


@dataclass(frozen=True)
class ConversationMetrics:
    """Immutable metrics for a conversation."""
    total_messages: int = 0
    user_messages: int = 0
    agent_messages: int = 0
    system_messages: int = 0
    error_messages: int = 0
    
    total_tokens: int = 0
    user_tokens: int = 0
    agent_tokens: int = 0
    
    total_audio_duration: float = 0.0
    user_audio_duration: float = 0.0
    agent_audio_duration: float = 0.0
    
    average_response_time: float = 0.0
    function_calls_count: int = 0
    response_time_count: int = 0

    def add_message(self, message: Message) -> 'ConversationMetrics':
        """Create new metrics with added message."""
        new_total = self.total_messages + 1
        new_user = self.user_messages + (1 if message.is_user_message() else 0)
        new_agent = self.agent_messages + (1 if message.is_agent_message() else 0)
        new_system = self.system_messages + (1 if message.message_type == MessageType.SYSTEM else 0)
        new_error = self.error_messages + (1 if message.message_type == MessageType.ERROR else 0)
        new_function_calls = self.function_calls_count + (1 if message.message_type == MessageType.FUNCTION_CALL else 0)
        
        new_total_tokens = self.total_tokens + message.tokens_used
        new_user_tokens = self.user_tokens + (message.tokens_used if message.is_user_message() else 0)
        new_agent_tokens = self.agent_tokens + (message.tokens_used if message.is_agent_message() else 0)
        
        audio_duration = message.audio_duration or 0.0
        new_total_audio = self.total_audio_duration + audio_duration
        new_user_audio = self.user_audio_duration + (audio_duration if message.is_user_message() else 0)
        new_agent_audio = self.agent_audio_duration + (audio_duration if message.is_agent_message() else 0)
        
        return ConversationMetrics(
            total_messages=new_total,
            user_messages=new_user,
            agent_messages=new_agent,
            system_messages=new_system,
            error_messages=new_error,
            total_tokens=new_total_tokens,
            user_tokens=new_user_tokens,
            agent_tokens=new_agent_tokens,
            total_audio_duration=new_total_audio,
            user_audio_duration=new_user_audio,
            agent_audio_duration=new_agent_audio,
            average_response_time=self.average_response_time,  # Will be calculated separately
            function_calls_count=new_function_calls
        )
    
    def update_response_time(self, new_response_time: float) -> 'ConversationMetrics':
        """Create new metrics with updated average response time."""
        # Calculate new average - this should be called AFTER adding the agent message
        new_count = self.response_time_count + 1
        total_time = self.average_response_time * self.response_time_count
        new_average = (total_time + new_response_time) / new_count
        
        return ConversationMetrics(
            total_messages=self.total_messages,
            user_messages=self.user_messages,
            agent_messages=self.agent_messages,
            system_messages=self.system_messages,
            error_messages=self.error_messages,
            total_tokens=self.total_tokens,
            user_tokens=self.user_tokens,
            agent_tokens=self.agent_tokens,
            total_audio_duration=self.total_audio_duration,
            user_audio_duration=self.user_audio_duration,
            agent_audio_duration=self.agent_audio_duration,
            average_response_time=new_average,
            function_calls_count=self.function_calls_count,
            response_time_count=new_count
        )

@dataclass
class Conversation:
    """
    Represents a conversation session with comprehensive state management.
    
    This entity encapsulates all the business logic related to a conversation,
    including message management, state transitions, and metrics.
    """
    
    # Core identifiers
    id: str
    client_id: str
    agent_id: Optional[str] = None
    
    # Conversation metadata
    title: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # State management
    state: ConversationState = ConversationState.INITIALIZED
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Messages and metrics
    messages: List[Message] = field(default_factory=list)
    metrics: ConversationMetrics = field(default_factory=ConversationMetrics)
    
    # Configuration
    max_messages: int = 1000
    max_context_tokens: int = 4000
    auto_summarize_threshold: int = 800
    
    # Internal state
    _last_user_message_time: Optional[datetime] = field(default=None, repr=False)
    _last_agent_message_time: Optional[datetime] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.id:
            raise ValueError("id cannot be empty")
        if not self.client_id:
            raise ValueError("client_id cannot be empty")
        
        # Ensure timestamps are timezone-aware
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if self.updated_at.tzinfo is None:
            self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        if len(self.messages) >= self.max_messages:
            raise ValueError(f"Maximum message limit ({self.max_messages}) exceeded")
        
        if message.conversation_id != self.id:
            raise ValueError("Message conversation_id does not match conversation id")
        
        # Update internal timestamps
        if message.is_user_message():
            self._last_user_message_time = message.timestamp
        elif message.is_agent_message():
            self._last_agent_message_time = message.timestamp
        
        # Add message and update metrics
        self.messages.append(message)
        self.metrics = self.metrics.add_message(message)
        
        # Update conversation state and timestamp
        self._update_state_from_message(message)
        self.updated_at = datetime.now(timezone.utc)
        
        # Auto-summarize if needed
        if self._should_auto_summarize():
            self._auto_summarize()
    
    def add_user_text_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a user text message and return it."""
        message = Message.create_user_text(
            conversation_id=self.id,
            content=content,
            client_id=self.client_id,
            metadata=metadata
        )
        self.add_message(message)
        return message
    
    def add_user_audio_message(
        self,
        content: str,
        audio_duration: float,
        audio_format: str = "webm",
        transcription_confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add a user audio message and return it."""
        message = Message.create_user_audio(
            conversation_id=self.id,
            content=content,
            client_id=self.client_id,
            audio_duration=audio_duration,
            audio_format=audio_format,
            transcription_confidence=transcription_confidence,
            metadata=metadata
        )
        self.add_message(message)
        return message
    
    def add_agent_response(
        self,
        content: str,
        message_type: MessageType = MessageType.AGENT_TEXT,
        tokens_used: int = 0,
        processing_time: float = 0.0,
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add an agent response message and return it."""
        if not self.agent_id:
            raise ValueError("Agent ID must be set before adding agent responses")
        
        message = Message.create_agent_response(
            conversation_id=self.id,
            content=content,
            agent_id=self.agent_id,
            message_type=message_type,
            tokens_used=tokens_used,
            processing_time=processing_time,
            confidence_score=confidence_score,
            metadata=metadata
        )
        self.add_message(message)
        
        # Update response time metrics
        if processing_time > 0:
            self.metrics = self.metrics.update_response_time(processing_time)
        
        return message
    
    def add_function_call(
        self,
        function_name: str,
        function_args: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add a function call message and return it."""
        if not self.agent_id:
            raise ValueError("Agent ID must be set before adding function calls")
        
        message = Message.create_function_call(
            conversation_id=self.id,
            function_name=function_name,
            function_args=function_args,
            agent_id=self.agent_id,
            metadata=metadata
        )
        self.add_message(message)
        return message
    
    def add_function_response(
        self,
        function_name: str,
        function_result: Any,
        processing_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add a function response message and return it."""
        message = Message.create_function_response(
            conversation_id=self.id,
            function_name=function_name,
            function_result=function_result,
            processing_time=processing_time,
            metadata=metadata
        )
        self.add_message(message)
        return message
    
    def add_system_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a system message and return it."""
        message = Message.create_system_message(
            conversation_id=self.id,
            content=content,
            metadata=metadata
        )
        self.add_message(message)
        return message
    
    def add_error_message(
        self,
        error_code: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add an error message and return it."""
        message = Message.create_error_message(
            conversation_id=self.id,
            error_code=error_code,
            error_message=error_message,
            error_details=error_details,
            metadata=metadata
        )
        self.add_message(message)
        return message
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get the most recent messages."""
        return self.messages[-count:] if count > 0 else []
    
    def get_messages_by_type(self, message_type: MessageType) -> List[Message]:
        """Get all messages of a specific type."""
        return [msg for msg in self.messages if msg.message_type == message_type]
    
    def get_user_messages(self) -> List[Message]:
        """Get all user messages."""
        return [msg for msg in self.messages if msg.is_user_message()]
    
    def get_agent_messages(self) -> List[Message]:
        """Get all agent messages."""
        return [msg for msg in self.messages if msg.is_agent_message()]
    
    def get_context_window(self, max_tokens: Optional[int] = None) -> List[Message]:
        """Get messages that fit within the context window."""
        max_tokens = max_tokens or self.max_context_tokens
        
        # Start from the end and work backwards
        context_messages = []
        total_tokens = 0
        
        for message in reversed(self.messages):
            if total_tokens + message.tokens_used > max_tokens:
                break
            
            context_messages.insert(0, message)
            total_tokens += message.tokens_used
        
        return context_messages
    
    def set_agent(self, agent_id: str) -> None:
        """Set the agent for this conversation."""
        self.agent_id = agent_id
        self.updated_at = datetime.now(timezone.utc)
    
    def update_context(self, key: str, value: Any) -> None:
        """Update conversation context."""
        self.context[key] = value
        self.updated_at = datetime.now(timezone.utc)
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get value from conversation context."""
        return self.context.get(key, default)
    
    def clear_context(self) -> None:
        """Clear conversation context."""
        self.context.clear()
        self.updated_at = datetime.now(timezone.utc)
    
    def activate(self) -> None:
        """Activate the conversation."""
        if not self.can_transition_to(ConversationState.ACTIVE):
            raise ValueError(f"Cannot activate conversation from state: {self.state}")
        
        self.state = ConversationState.ACTIVE
        self.updated_at = datetime.now(timezone.utc)
    
    def pause(self) -> None:
        """Pause the conversation."""
        if not self.can_transition_to(ConversationState.PAUSED):
            raise ValueError(f"Cannot pause conversation from state: {self.state}")
        
        self.state = ConversationState.PAUSED
        self.updated_at = datetime.now(timezone.utc)
    
    def resume(self) -> None:
        """Resume a paused conversation."""
        if self.state != ConversationState.PAUSED:
            raise ValueError("Can only resume paused conversations")
        
        self.state = ConversationState.ACTIVE
        self.updated_at = datetime.now(timezone.utc)
    
    def complete(self) -> None:
        """Mark conversation as completed."""
        if not self.can_transition_to(ConversationState.COMPLETED):
            raise ValueError(f"Cannot complete conversation from state: {self.state}")
        
        self.state = ConversationState.COMPLETED
        self.updated_at = datetime.now(timezone.utc)
    
    def terminate(self, reason: Optional[str] = None) -> None:
        """Terminate the conversation."""
        self.state = ConversationState.TERMINATED
        self.updated_at = datetime.now(timezone.utc)
        
        if reason:
            self.add_system_message(f"Conversation terminated: {reason}")
    
    def mark_error(self, error_message: str) -> None:
        """Mark conversation as in error state."""
        self.state = ConversationState.ERROR
        self.updated_at = datetime.now(timezone.utc)
        self.add_error_message("CONVERSATION_ERROR", error_message)
    
    def can_transition_to(self, new_state: ConversationState) -> bool:
        """Check if transition to new state is valid."""
        valid_transitions = {
            ConversationState.INITIALIZED: [
                ConversationState.ACTIVE,
                ConversationState.TERMINATED,
                ConversationState.ERROR
            ],
            ConversationState.ACTIVE: [
                ConversationState.WAITING_FOR_INPUT,
                ConversationState.PROCESSING,
                ConversationState.PAUSED,
                ConversationState.COMPLETED,
                ConversationState.TERMINATED,
                ConversationState.ERROR
            ],
            ConversationState.WAITING_FOR_INPUT: [
                ConversationState.ACTIVE,
                ConversationState.PROCESSING,
                ConversationState.PAUSED,
                ConversationState.TERMINATED,
                ConversationState.ERROR
            ],
            ConversationState.PROCESSING: [
                ConversationState.ACTIVE,
                ConversationState.WAITING_FOR_INPUT,
                ConversationState.ERROR
            ],
            ConversationState.PAUSED: [
                ConversationState.ACTIVE,
                ConversationState.TERMINATED,
                ConversationState.ERROR
            ],
            ConversationState.COMPLETED: [
                ConversationState.TERMINATED
            ],
            ConversationState.TERMINATED: [],  # Final state
            ConversationState.ERROR: [
                ConversationState.ACTIVE,
                ConversationState.TERMINATED
            ]
        }
        
        return new_state in valid_transitions.get(self.state, [])
    
    def get_conversation_duration(self) -> float:
        """Get conversation duration in seconds."""
        return (self.updated_at - self.created_at).total_seconds()
    
    def get_last_activity_time(self) -> datetime:
        """Get time of last activity (message)."""
        return self.updated_at
    
    def get_response_time_stats(self) -> Dict[str, float]:
        """Get response time statistics."""
        if not self._last_user_message_time or not self._last_agent_message_time:
            return {"average": 0.0, "count": 0}
        
        # Calculate response times between user and agent messages
        response_times = []
        user_messages = self.get_user_messages()
        agent_messages = self.get_agent_messages()
        
        for user_msg in user_messages:
            # Find the next agent message after this user message
            next_agent_msg = next(
                (agent_msg for agent_msg in agent_messages 
                 if agent_msg.timestamp > user_msg.timestamp),
                None
            )
            if next_agent_msg:
                response_time = (next_agent_msg.timestamp - user_msg.timestamp).total_seconds()
                response_times.append(response_time)
        
        if not response_times:
            return {"average": 0.0, "count": 0}
        
        return {
            "average": sum(response_times) / len(response_times),
            "min": min(response_times),
            "max": max(response_times),
            "count": len(response_times)
        }
    
    def _update_state_from_message(self, message: Message) -> None:
        """Update conversation state based on message type."""
        if message.message_type == MessageType.ERROR:
            self.state = ConversationState.ERROR
        elif message.is_user_message() and self.state == ConversationState.WAITING_FOR_INPUT:
            self.state = ConversationState.PROCESSING
        elif message.is_agent_message() and self.state == ConversationState.PROCESSING:
            self.state = ConversationState.WAITING_FOR_INPUT
        elif self.state == ConversationState.INITIALIZED:
            self.state = ConversationState.ACTIVE
    
    def _should_auto_summarize(self) -> bool:
        """Check if conversation should be auto-summarized."""
        return (
            len(self.messages) >= self.auto_summarize_threshold and
            self.metrics.total_tokens >= self.max_context_tokens * 0.8
        )
    
    def _auto_summarize(self) -> None:
        """Auto-summarize old messages to maintain context window."""
        # This is a placeholder for summarization logic
        # In practice, this would call an AI service to summarize old messages
        if len(self.messages) > self.auto_summarize_threshold:
            # Keep recent messages and summarize older ones
            recent_messages = self.messages[-50:]  # Keep last 50 messages
            older_messages = self.messages[:-50]
            
            # Create summary message (simplified)
            summary_content = f"Summary of {len(older_messages)} previous messages"
            summary_message = Message.create_system_message(
                conversation_id=self.id,
                content=summary_content,
                metadata={"type": "auto_summary", "message_count": len(older_messages)}
            )
            
            # Replace old messages with summary
            self.messages = [summary_message] + recent_messages
            
            # Recalculate metrics (simplified)
            self.metrics = ConversationMetrics()
            for msg in self.messages:
                self.metrics = self.metrics.add_message(msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary for serialization."""
        return {
            "id": self.id,
            "client_id": self.client_id,
            "agent_id": self.agent_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "state": self.state.value,
            "context": self.context,
            "message_count": len(self.messages),
            "metrics": {
                "total_messages": self.metrics.total_messages,
                "user_messages": self.metrics.user_messages,
                "agent_messages": self.metrics.agent_messages,
                "system_messages": self.metrics.system_messages,
                "error_messages": self.metrics.error_messages,
                "total_tokens": self.metrics.total_tokens,
                "user_tokens": self.metrics.user_tokens,
                "agent_tokens": self.metrics.agent_tokens,
                "total_audio_duration": self.metrics.total_audio_duration,
                "user_audio_duration": self.metrics.user_audio_duration,
                "agent_audio_duration": self.metrics.agent_audio_duration,
                "average_response_time": self.metrics.average_response_time,
                "function_calls_count": self.metrics.function_calls_count
            },
            "duration": self.get_conversation_duration(),
            "response_time_stats": self.get_response_time_stats()
        }
    
    def __str__(self) -> str:
        return f"Conversation(id={self.id}, state={self.state.value}, messages={len(self.messages)})"