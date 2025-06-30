"""
Comprehensive tests for the Conversation and Message entities.

These tests cover all business logic, state transitions, metrics,
and edge cases for the conversation domain entities.
"""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from src.core.entities.conversation import (
    Conversation,
    Message,
    MessageType,
    ConversationState,
    ConversationMetrics
)


class TestMessage:
    """Test the immutable Message class."""
    
    def test_message_creation_with_validation(self):
        """Test message creation with validation."""
        # Valid message
        message = Message(
            id="msg-123",
            conversation_id="conv-123",
            message_type=MessageType.USER_TEXT,
            content="Hello world",
            timestamp=datetime.now(timezone.utc)
        )
        
        assert message.id == "msg-123"
        assert message.conversation_id == "conv-123"
        assert message.content == "Hello world"
    
    def test_message_validation_errors(self):
        """Test message validation errors."""
        # Empty conversation_id
        with pytest.raises(ValueError, match="conversation_id cannot be empty"):
            Message(
                id="msg-123",
                conversation_id="",
                message_type=MessageType.USER_TEXT,
                content="Hello",
                timestamp=datetime.now(timezone.utc)
            )
        
        # Empty content for non-function message
        with pytest.raises(ValueError, match="content cannot be empty"):
            Message(
                id="msg-123",
                conversation_id="conv-123",
                message_type=MessageType.USER_TEXT,
                content="",
                timestamp=datetime.now(timezone.utc)
            )
    
    def test_message_auto_id_generation(self):
        """Test automatic ID generation when not provided."""
        message = Message(
            id="",
            conversation_id="conv-123",
            message_type=MessageType.USER_TEXT,
            content="Hello",
            timestamp=datetime.now(timezone.utc)
        )
        
        assert message.id != ""
        assert len(message.id) > 0
    
    def test_message_timezone_handling(self):
        """Test timezone handling for timestamps."""
        naive_time = datetime.now()
        message = Message(
            id="msg-123",
            conversation_id="conv-123",
            message_type=MessageType.USER_TEXT,
            content="Hello",
            timestamp=naive_time
        )
        
        assert message.timestamp.tzinfo == timezone.utc
    
    def test_create_user_text_message(self):
        """Test creating user text message."""
        message = Message.create_user_text(
            conversation_id="conv-123",
            content="Hello AI",
            client_id="client-456",
            metadata={"platform": "web"}
        )
        
        assert message.message_type == MessageType.USER_TEXT
        assert message.content == "Hello AI"
        assert message.client_id == "client-456"
        assert message.metadata == {"platform": "web"}
        assert message.is_user_message() is True
        assert message.is_agent_message() is False
    
    def test_create_user_audio_message(self):
        """Test creating user audio message."""
        message = Message.create_user_audio(
            conversation_id="conv-123",
            content="Hello AI",
            client_id="client-456",
            audio_duration=5.5,
            audio_format="webm",
            transcription_confidence=0.95
        )
        
        assert message.message_type == MessageType.USER_AUDIO
        assert message.audio_duration == 5.5
        assert message.audio_format == "webm"
        assert message.transcription_confidence == 0.95
        assert message.has_audio_data() is True
    
    def test_create_agent_response(self):
        """Test creating agent response message."""
        message = Message.create_agent_response(
            conversation_id="conv-123",
            content="Hello human",
            agent_id="agent-789",
            tokens_used=50,
            processing_time=1.2,
            confidence_score=0.98
        )
        
        assert message.message_type == MessageType.AGENT_TEXT
        assert message.agent_id == "agent-789"
        assert message.tokens_used == 50
        assert message.processing_time == 1.2
        assert message.confidence_score == 0.98
        assert message.is_agent_message() is True
    
    def test_create_function_call(self):
        """Test creating function call message."""
        args = {"query": "weather in NYC"}
        message = Message.create_function_call(
            conversation_id="conv-123",
            function_name="get_weather",
            function_args=args,
            agent_id="agent-789"
        )
        
        assert message.message_type == MessageType.FUNCTION_CALL
        assert message.function_name == "get_weather"
        assert message.function_args == args
        assert message.is_function_related() is True
    
    def test_create_function_response(self):
        """Test creating function response message."""
        result = {"temperature": 72, "condition": "sunny"}
        message = Message.create_function_response(
            conversation_id="conv-123",
            function_name="get_weather",
            function_result=result,
            processing_time=0.5
        )
        
        assert message.message_type == MessageType.FUNCTION_RESPONSE
        assert message.function_result == result
        assert message.processing_time == 0.5
        assert message.is_function_related() is True
    
    def test_create_system_message(self):
        """Test creating system message."""
        message = Message.create_system_message(
            conversation_id="conv-123",
            content="Session started"
        )
        
        assert message.message_type == MessageType.SYSTEM
        assert message.content == "Session started"
    
    def test_create_error_message(self):
        """Test creating error message."""
        error_details = {"code": 500, "trace": "stack trace"}
        message = Message.create_error_message(
            conversation_id="conv-123",
            error_code="PROCESSING_ERROR",
            error_message="Failed to process request",
            error_details=error_details
        )
        
        assert message.message_type == MessageType.ERROR
        assert message.error_code == "PROCESSING_ERROR"
        assert message.content == "Failed to process request"
        assert message.error_details == error_details
    
    def test_message_serialization(self):
        """Test message to dictionary conversion."""
        message = Message.create_user_text(
            conversation_id="conv-123",
            content="Hello",
            client_id="client-456",
            metadata={"test": True}
        )
        
        message_dict = message.to_dict()
        
        assert message_dict["id"] == message.id
        assert message_dict["conversation_id"] == "conv-123"
        assert message_dict["message_type"] == MessageType.USER_TEXT.value
        assert message_dict["content"] == "Hello"
        assert message_dict["client_id"] == "client-456"
        assert message_dict["metadata"] == {"test": True}


class TestConversationMetrics:
    """Test the immutable ConversationMetrics class."""
    
    def test_default_metrics(self):
        """Test default metrics initialization."""
        metrics = ConversationMetrics()
        
        assert metrics.total_messages == 0
        assert metrics.user_messages == 0
        assert metrics.agent_messages == 0
        assert metrics.total_tokens == 0
        assert metrics.total_audio_duration == 0.0
        assert metrics.average_response_time == 0.0
    
    def test_add_user_message(self):
        """Test adding user message to metrics."""
        metrics = ConversationMetrics()
        message = Message.create_user_text(
            conversation_id="conv-123",
            content="Hello",
            client_id="client-456"
        )
        
        new_metrics = metrics.add_message(message)
        
        assert metrics.total_messages == 0  # Original unchanged
        assert new_metrics.total_messages == 1
        assert new_metrics.user_messages == 1
        assert new_metrics.agent_messages == 0
    
    def test_add_agent_message_with_tokens(self):
        """Test adding agent message with tokens."""
        metrics = ConversationMetrics()
        message = Message.create_agent_response(
            conversation_id="conv-123",
            content="Hello",
            agent_id="agent-789",
            tokens_used=100
        )
        
        new_metrics = metrics.add_message(message)
        
        assert new_metrics.agent_messages == 1
        assert new_metrics.total_tokens == 100
        assert new_metrics.agent_tokens == 100
        assert new_metrics.user_tokens == 0
    
    def test_add_audio_message(self):
        """Test adding audio message to metrics."""
        metrics = ConversationMetrics()
        message = Message.create_user_audio(
            conversation_id="conv-123",
            content="Hello",
            client_id="client-456",
            audio_duration=5.5
        )
        
        new_metrics = metrics.add_message(message)
        
        assert new_metrics.total_audio_duration == 5.5
        assert new_metrics.user_audio_duration == 5.5
        assert new_metrics.agent_audio_duration == 0.0
    
    def test_update_response_time(self):
        """Test updating response time."""
        metrics = ConversationMetrics()
        
        # First response time
        metrics = metrics.update_response_time(2.0)
        assert metrics.average_response_time == 2.0
        
        # Second response time
        metrics = metrics.update_response_time(4.0)
        assert metrics.average_response_time == 3.0  # (2.0 + 4.0) / 2


class TestConversation:
    """Test the Conversation entity."""
    
    def test_conversation_creation(self):
        """Test basic conversation creation."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        assert conv.id == "conv-123"
        assert conv.client_id == "client-456"
        assert conv.state == ConversationState.INITIALIZED
        assert len(conv.messages) == 0
        assert isinstance(conv.metrics, ConversationMetrics)
    
    def test_conversation_validation(self):
        """Test conversation validation."""
        # Empty ID
        with pytest.raises(ValueError, match="id cannot be empty"):
            Conversation(id="", client_id="client-456")
        
        # Empty client_id
        with pytest.raises(ValueError, match="client_id cannot be empty"):
            Conversation(id="conv-123", client_id="")
    
    def test_add_user_text_message(self):
        """Test adding user text message."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        message = conv.add_user_text_message("Hello AI", {"platform": "web"})
        
        assert len(conv.messages) == 1
        assert message.content == "Hello AI"
        assert message.message_type == MessageType.USER_TEXT
        assert conv.metrics.user_messages == 1
        assert conv.state == ConversationState.ACTIVE
    
    def test_add_agent_response(self):
        """Test adding agent response."""
        conv = Conversation(id="conv-123", client_id="client-456", agent_id="agent-789")
        
        message = conv.add_agent_response(
            content="Hello human",
            tokens_used=50,
            processing_time=1.5
        )
        
        assert len(conv.messages) == 1
        assert message.content == "Hello human"
        assert message.tokens_used == 50
        assert conv.metrics.agent_messages == 1
        assert conv.metrics.total_tokens == 50
    
    def test_add_agent_response_without_agent_id(self):
        """Test adding agent response without agent ID should fail."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        with pytest.raises(ValueError, match="Agent ID must be set"):
            conv.add_agent_response("Hello human")
    
    def test_add_user_audio_message(self):
        """Test adding user audio message."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        message = conv.add_user_audio_message(
            content="Hello AI",
            audio_duration=5.5,
            audio_format="webm",
            transcription_confidence=0.95
        )
        
        assert message.message_type == MessageType.USER_AUDIO
        assert message.audio_duration == 5.5
        assert conv.metrics.total_audio_duration == 5.5
        assert conv.metrics.user_audio_duration == 5.5
    
    def test_add_function_call_and_response(self):
        """Test adding function call and response."""
        conv = Conversation(id="conv-123", client_id="client-456", agent_id="agent-789")
        
        # Add function call
        call_message = conv.add_function_call(
            function_name="get_weather",
            function_args={"city": "NYC"}
        )
        
        # Add function response
        response_message = conv.add_function_response(
            function_name="get_weather",
            function_result={"temperature": 72},
            processing_time=0.5
        )
        
        assert len(conv.messages) == 2
        assert call_message.function_name == "get_weather"
        assert response_message.function_result == {"temperature": 72}
        assert conv.metrics.function_calls_count == 1
    
    def test_add_system_and_error_messages(self):
        """Test adding system and error messages."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        # Add system message
        system_msg = conv.add_system_message("Session started")
        
        # Add error message
        error_msg = conv.add_error_message(
            error_code="TIMEOUT",
            error_message="Request timed out",
            error_details={"timeout": 30}
        )
        
        assert len(conv.messages) == 2
        assert system_msg.message_type == MessageType.SYSTEM
        assert error_msg.message_type == MessageType.ERROR
        assert conv.metrics.system_messages == 1
        assert conv.metrics.error_messages == 1
        assert conv.state == ConversationState.ERROR  # Error message changes state
    
    def test_message_validation_in_conversation(self):
        """Test message validation when adding to conversation."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        # Create message with wrong conversation_id
        wrong_message = Message.create_user_text(
            conversation_id="wrong-conv-id",
            content="Hello",
            client_id="client-456"
        )
        
        with pytest.raises(ValueError, match="Message conversation_id does not match"):
            conv.add_message(wrong_message)
    
    def test_max_messages_limit(self):
        """Test maximum messages limit."""
        conv = Conversation(id="conv-123", client_id="client-456")
        conv.max_messages = 2
        
        # Add messages up to limit
        conv.add_user_text_message("Message 1")
        conv.add_user_text_message("Message 2")
        
        # Third message should fail
        with pytest.raises(ValueError, match="Maximum message limit"):
            conv.add_user_text_message("Message 3")
    
    def test_conversation_state_transitions(self):
        """Test conversation state transitions."""
        conv = Conversation(id="conv-123", client_id="client-456", agent_id="agent-789")
        
        # Test valid transitions
        assert conv.can_transition_to(ConversationState.ACTIVE)
        conv.activate()
        assert conv.state == ConversationState.ACTIVE
        
        assert conv.can_transition_to(ConversationState.PAUSED)
        conv.pause()
        assert conv.state == ConversationState.PAUSED
        
        conv.resume()
        assert conv.state == ConversationState.ACTIVE
        
        conv.complete()
        assert conv.state == ConversationState.COMPLETED
    
    def test_invalid_state_transitions(self):
        """Test invalid state transitions."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        # Cannot complete from initialized state
        with pytest.raises(ValueError, match="Cannot complete conversation"):
            conv.complete()
        
        # Cannot resume non-paused conversation
        with pytest.raises(ValueError, match="Can only resume paused"):
            conv.resume()
    
    def test_conversation_termination(self):
        """Test conversation termination."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        conv.terminate("User disconnected")
        
        assert conv.state == ConversationState.TERMINATED
        assert len(conv.messages) == 1
        assert conv.messages[0].message_type == MessageType.SYSTEM
        assert "User disconnected" in conv.messages[0].content
    
    def test_conversation_error_handling(self):
        """Test conversation error handling."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        conv.mark_error("Processing failed")
        
        assert conv.state == ConversationState.ERROR
        assert len(conv.messages) == 1
        assert conv.messages[0].message_type == MessageType.ERROR
    
    def test_get_recent_messages(self):
        """Test getting recent messages."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        # Add multiple messages
        for i in range(5):
            conv.add_user_text_message(f"Message {i}")
        
        recent = conv.get_recent_messages(3)
        assert len(recent) == 3
        assert recent[0].content == "Message 2"
        assert recent[2].content == "Message 4"
    
    def test_get_messages_by_type(self):
        """Test getting messages by type."""
        conv = Conversation(id="conv-123", client_id="client-456", agent_id="agent-789")
        
        conv.add_user_text_message("User message")
        conv.add_agent_response("Agent response")
        conv.add_system_message("System message")
        
        user_messages = conv.get_messages_by_type(MessageType.USER_TEXT)
        agent_messages = conv.get_messages_by_type(MessageType.AGENT_TEXT)
        system_messages = conv.get_messages_by_type(MessageType.SYSTEM)
        
        assert len(user_messages) == 1
        assert len(agent_messages) == 1
        assert len(system_messages) == 1
    
    def test_get_user_and_agent_messages(self):
        """Test getting user and agent messages."""
        conv = Conversation(id="conv-123", client_id="client-456", agent_id="agent-789")
        
        conv.add_user_text_message("User text")
        conv.add_user_audio_message("User audio", 5.0)
        conv.add_agent_response("Agent response")
        conv.add_system_message("System message")
        
        user_messages = conv.get_user_messages()
        agent_messages = conv.get_agent_messages()
        
        assert len(user_messages) == 2
        assert len(agent_messages) == 1
        assert all(msg.is_user_message() for msg in user_messages)
        assert all(msg.is_agent_message() for msg in agent_messages)
    
    def test_context_window(self):
        """Test getting messages within context window."""
        conv = Conversation(id="conv-123", client_id="client-456", agent_id="agent-789")
        
        # Add messages with tokens
        conv.add_user_text_message("Message 1")
        conv.messages[-1] = conv.messages[-1].__class__(
            **{**conv.messages[-1].__dict__, 'tokens_used': 100}
        )
        
        conv.add_agent_response("Response 1", tokens_used=150)
        conv.add_user_text_message("Message 2") 
        conv.messages[-1] = conv.messages[-1].__class__(
            **{**conv.messages[-1].__dict__, 'tokens_used': 200}
        )
        
        # Get context window with limit
        context = conv.get_context_window(max_tokens=300)
        
        # Should include messages that fit within 300 tokens
        total_tokens = sum(msg.tokens_used for msg in context)
        assert total_tokens <= 300
        assert len(context) <= 3
    
    def test_agent_management(self):
        """Test agent management."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        assert conv.agent_id is None
        
        conv.set_agent("agent-789")
        assert conv.agent_id == "agent-789"
    
    def test_context_management(self):
        """Test conversation context management."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        # Update context
        conv.update_context("user_preference", "voice")
        conv.update_context("language", "en")
        
        assert conv.get_context("user_preference") == "voice"
        assert conv.get_context("language") == "en"
        assert conv.get_context("nonexistent", "default") == "default"
        
        # Clear context
        conv.clear_context()
        assert len(conv.context) == 0
    
    def test_conversation_duration(self):
        """Test conversation duration calculation."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        # Set created_at to 1 hour ago
        conv.created_at = datetime.now(timezone.utc) - timedelta(hours=1)
        conv.updated_at = datetime.now(timezone.utc)
        
        duration = conv.get_conversation_duration()
        assert duration >= 3600  # At least 1 hour in seconds
    
    def test_response_time_stats(self):
        """Test response time statistics."""
        conv = Conversation(id="conv-123", client_id="client-456", agent_id="agent-789")
        
        # Create messages with specific timestamps
        base_time = datetime.now(timezone.utc)
        
        # User message at base time
        user_msg = conv.add_user_text_message("Hello")
        user_msg.__dict__['timestamp'] = base_time
        
        # Agent response 2 seconds later
        agent_msg = conv.add_agent_response("Hi there")
        agent_msg.__dict__['timestamp'] = base_time + timedelta(seconds=2)
        
        conv._last_user_message_time = base_time
        conv._last_agent_message_time = base_time + timedelta(seconds=2)
        
        stats = conv.get_response_time_stats()
        assert stats["count"] >= 0  # May be 0 due to timing logic
    
    def test_auto_summarization_threshold(self):
        """Test auto-summarization threshold detection."""
        conv = Conversation(id="conv-123", client_id="client-456")
        conv.auto_summarize_threshold = 3
        conv.max_context_tokens = 100
        
        # Add messages to exceed threshold
        conv.add_user_text_message("Message 1")
        conv.add_user_text_message("Message 2") 
        conv.add_user_text_message("Message 3")
        
        # Manually set tokens to exceed threshold (80% of 100 = 80)
        conv.metrics = conv.metrics.__class__(
            **{**conv.metrics.__dict__, 'total_tokens': 85}
        )
        
        # Should trigger auto-summarization internally
        assert conv._should_auto_summarize() is True
    
    def test_conversation_serialization(self):
        """Test conversation to dictionary conversion."""
        conv = Conversation(id="conv-123", client_id="client-456", agent_id="agent-789")
        conv.title = "Test Conversation"
        conv.update_context("test_key", "test_value")
        conv.add_user_text_message("Hello")
        
        conv_dict = conv.to_dict()
        
        assert conv_dict["id"] == "conv-123"
        assert conv_dict["client_id"] == "client-456"
        assert conv_dict["agent_id"] == "agent-789"
        assert conv_dict["title"] == "Test Conversation"
        assert conv_dict["state"] == ConversationState.ACTIVE.value
        assert conv_dict["context"] == {"test_key": "test_value"}
        assert conv_dict["message_count"] == 1
        assert "metrics" in conv_dict
        assert "duration" in conv_dict
        assert "response_time_stats" in conv_dict
    
    def test_string_representation(self):
        """Test string representation of conversation."""
        conv = Conversation(id="conv-123", client_id="client-456")
        conv.add_user_text_message("Hello")
        
        str_repr = str(conv)
        
        assert "conv-123" in str_repr
        assert ConversationState.ACTIVE.value in str_repr
        assert "messages=1" in str_repr
    
    def test_timezone_handling_in_conversation(self):
        """Test proper timezone handling in conversation."""
        # Test with naive datetime
        naive_time = datetime.now()
        conv = Conversation(
            id="conv-123",
            client_id="client-456",
            created_at=naive_time,
            updated_at=naive_time
        )
        
        # Should be converted to UTC
        assert conv.created_at.tzinfo == timezone.utc
        assert conv.updated_at.tzinfo == timezone.utc
    
    def test_message_ordering_and_timestamps(self):
        """Test that messages maintain proper ordering and timestamps."""
        conv = Conversation(id="conv-123", client_id="client-456", agent_id="agent-789")
        
        # Add messages with small delays to ensure different timestamps
        import time
        
        msg1 = conv.add_user_text_message("First message")
        time.sleep(0.01)
        msg2 = conv.add_agent_response("Second message")
        time.sleep(0.01)
        msg3 = conv.add_user_text_message("Third message")
        
        assert msg1.timestamp < msg2.timestamp < msg3.timestamp
        assert conv.messages[0] == msg1
        assert conv.messages[1] == msg2
        assert conv.messages[2] == msg3
    
    def test_conversation_state_from_messages(self):
        """Test conversation state updates from message types."""
        conv = Conversation(id="conv-123", client_id="client-456", agent_id="agent-789")
        
        # Initial state
        assert conv.state == ConversationState.INITIALIZED
        
        # User message should activate conversation
        conv.add_user_text_message("Hello")
        assert conv.state == ConversationState.ACTIVE
        
        # Error message should set error state
        conv.add_error_message("ERROR", "Something went wrong")
        assert conv.state == ConversationState.ERROR
    
    def test_function_call_workflow(self):
        """Test complete function call workflow."""
        conv = Conversation(id="conv-123", client_id="client-456", agent_id="agent-789")
        
        # User asks for weather
        conv.add_user_text_message("What's the weather in NYC?")
        
        # Agent makes function call
        call_msg = conv.add_function_call(
            function_name="get_weather",
            function_args={"city": "NYC", "units": "fahrenheit"}
        )
        
        # Function returns result
        response_msg = conv.add_function_response(
            function_name="get_weather",
            function_result={"temperature": 72, "condition": "sunny", "humidity": 65},
            processing_time=1.2
        )
        
        # Agent responds with formatted result
        agent_msg = conv.add_agent_response(
            "The weather in NYC is 72°F and sunny with 65% humidity.",
            tokens_used=25
        )
        
        assert len(conv.messages) == 4
        assert conv.metrics.function_calls_count == 1
        assert conv.metrics.total_tokens == 25
        
        # Verify function call details
        assert call_msg.function_args["city"] == "NYC"
        assert response_msg.function_result["temperature"] == 72
        assert response_msg.processing_time == 1.2
    
    def test_conversation_edge_cases(self):
        """Test conversation edge cases and error conditions."""
        conv = Conversation(id="conv-123", client_id="client-456")
        
        # Get recent messages when no messages exist
        recent = conv.get_recent_messages(5)
        assert len(recent) == 0
        
        # Get context window when no messages exist
        context = conv.get_context_window(1000)
        assert len(context) == 0
        
        # Get response time stats when no agent messages exist
        stats = conv.get_response_time_stats()
        assert stats["average"] == 0.0
        assert stats["count"] == 0
    
    def test_metrics_consistency(self):
        """Test that metrics remain consistent with actual messages."""
        conv = Conversation(id="conv-123", client_id="client-456", agent_id="agent-789")
        
        # Add various types of messages
        conv.add_user_text_message("User message 1")
        conv.add_user_audio_message("User audio", 3.5)
        conv.add_agent_response("Agent response", tokens_used=50)
        conv.add_system_message("System message")
        conv.add_error_message("ERROR", "Error occurred")
        conv.add_function_call("test_function", {"arg": "value"})
        
        # Verify metrics match actual message counts
        user_msgs = conv.get_user_messages()
        agent_msgs = conv.get_agent_messages()
        system_msgs = conv.get_messages_by_type(MessageType.SYSTEM)
        error_msgs = conv.get_messages_by_type(MessageType.ERROR)
        function_calls = conv.get_messages_by_type(MessageType.FUNCTION_CALL)
        
        assert conv.metrics.user_messages == len(user_msgs)
        assert conv.metrics.agent_messages == len(agent_msgs)
        assert conv.metrics.system_messages == len(system_msgs)
        assert conv.metrics.error_messages == len(error_msgs)
        assert conv.metrics.function_calls_count == len(function_calls)
        assert conv.metrics.total_messages == len(conv.messages)
    
    def test_immutability_of_metrics(self):
        """Test that metrics objects are properly immutable."""
        conv = Conversation(id="conv-123", client_id="client-456")
        original_metrics = conv.metrics
        original_id = id(conv.metrics)
        
        # Add a message
        conv.add_user_text_message("Test message")
        
        # Metrics object should be different
        assert id(conv.metrics) != original_id
        assert original_metrics.total_messages == 0
        assert conv.metrics.total_messages == 1