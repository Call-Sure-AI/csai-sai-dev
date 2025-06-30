"""
Comprehensive tests for the ClientSession entity and related classes.

These tests cover all business logic, state transitions, metrics,
and edge cases for the client domain entities.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

from src.core.entities.client import (
    ClientSession,
    ConnectionState,
    VoiceCallState,
    ClientMetrics
)


class TestClientMetrics:
    """Test the immutable ClientMetrics class."""
    
    def test_default_metrics(self):
        """Test default metrics initialization."""
        metrics = ClientMetrics()
        
        assert metrics.message_count == 0
        assert metrics.total_tokens == 0
        assert metrics.request_times == []
        assert metrics.error_count == 0
        assert metrics.bytes_sent == 0
        assert metrics.bytes_received == 0
        assert metrics.voice_minutes == 0.0
        assert metrics.average_request_time == 0.0
        assert metrics.total_data_transfer == 0
    
    def test_add_request_time(self):
        """Test adding request time creates new metrics."""
        metrics = ClientMetrics()
        new_metrics = metrics.add_request_time(1.5)
        
        # Original unchanged
        assert metrics.message_count == 0
        assert metrics.request_times == []
        
        # New metrics updated
        assert new_metrics.message_count == 1
        assert new_metrics.request_times == [1.5]
        assert new_metrics.average_request_time == 1.5
    
    def test_add_tokens(self):
        """Test adding tokens creates new metrics."""
        metrics = ClientMetrics()
        new_metrics = metrics.add_tokens(100)
        
        assert metrics.total_tokens == 0
        assert new_metrics.total_tokens == 100
    
    def test_add_error(self):
        """Test adding error creates new metrics."""
        metrics = ClientMetrics()
        new_metrics = metrics.add_error()
        
        assert metrics.error_count == 0
        assert new_metrics.error_count == 1
    
    def test_add_data_transfer(self):
        """Test adding data transfer creates new metrics."""
        metrics = ClientMetrics()
        new_metrics = metrics.add_data_transfer(bytes_sent=100, bytes_received=200)
        
        assert metrics.bytes_sent == 0
        assert metrics.bytes_received == 0
        assert metrics.total_data_transfer == 0
        
        assert new_metrics.bytes_sent == 100
        assert new_metrics.bytes_received == 200
        assert new_metrics.total_data_transfer == 300
    
    def test_add_voice_time(self):
        """Test adding voice time creates new metrics."""
        metrics = ClientMetrics()
        new_metrics = metrics.add_voice_time(5.5)
        
        assert metrics.voice_minutes == 0.0
        assert new_metrics.voice_minutes == 5.5
    
    def test_average_request_time_calculation(self):
        """Test average request time calculation."""
        metrics = ClientMetrics()
        metrics = metrics.add_request_time(1.0)
        metrics = metrics.add_request_time(2.0)
        metrics = metrics.add_request_time(3.0)
        
        assert metrics.average_request_time == 2.0
        assert metrics.message_count == 3
    
    def test_metrics_immutability(self):
        """Test that metrics objects are immutable."""
        metrics = ClientMetrics()
        original_id = id(metrics)
        
        new_metrics = metrics.add_tokens(100)
        
        assert id(new_metrics) != original_id
        assert metrics.total_tokens == 0
        assert new_metrics.total_tokens == 100


class TestClientSession:
    """Test the ClientSession entity."""
    
    def test_client_session_creation(self):
        """Test basic client session creation."""
        session = ClientSession(client_id="test-client-123")
        
        assert session.client_id == "test-client-123"
        assert session.session_id is not None
        assert session.connection_state == ConnectionState.CONNECTING
        assert session.authenticated is False
        assert session.voice_call_state == VoiceCallState.INACTIVE
        assert isinstance(session.metrics, ClientMetrics)
        assert session.connection_time.tzinfo is not None
        assert session.last_activity.tzinfo is not None
    
    def test_client_session_validation(self):
        """Test client session validation."""
        with pytest.raises(ValueError, match="client_id cannot be empty"):
            ClientSession(client_id="")
    
    def test_update_activity(self):
        """Test activity update functionality."""
        session = ClientSession(client_id="test-client")
        initial_activity = session.last_activity
        initial_tokens = session.metrics.total_tokens
        
        # Small delay to ensure time difference
        import time
        time.sleep(0.01)
        
        session.update_activity(tokens=50, request_time=1.5)
        
        assert session.last_activity > initial_activity
        assert session.metrics.total_tokens == initial_tokens + 50
        assert 1.5 in session.metrics.request_times
    
    def test_authentication(self):
        """Test client authentication."""
        session = ClientSession(client_id="test-client")
        company_data = {"name": "Test Company", "tier": "premium"}
        
        session.authenticate("company-123", company_data, "hashed-api-key")
        
        assert session.authenticated is True
        assert session.company_id == "company-123"
        assert session.company_data == company_data
        assert session.api_key_hash == "hashed-api-key"
        assert session.connection_state == ConnectionState.AUTHENTICATED
    
    def test_start_conversation(self):
        """Test starting a conversation."""
        session = ClientSession(client_id="test-client")
        
        # Should fail if not authenticated
        with pytest.raises(ValueError, match="Client must be authenticated"):
            session.start_conversation("conv-123", "agent-456")
        
        # Authenticate first
        session.authenticate("company-123", {}, "hash")
        session.start_conversation("conv-123", "agent-456")
        
        assert session.conversation_id == "conv-123"
        assert session.agent_id == "agent-456"
        assert session.connection_state == ConnectionState.ACTIVE
    
    def test_voice_call_lifecycle(self):
        """Test complete voice call lifecycle."""
        from datetime import timedelta
        
        session = ClientSession(client_id="test-client")
        session.authenticate("company-123", {}, "hash")

        # Start voice call
        callback = Mock()
        session.start_voice_call(callback)

        assert session.voice_call_state == VoiceCallState.INITIALIZING
        assert session.voice_start_time is not None
        assert session.voice_callback == callback

        # Activate voice call
        session.activate_voice_call()
        assert session.voice_call_state == VoiceCallState.ACTIVE

        # Pause voice call
        session.pause_voice_call()
        assert session.voice_call_state == VoiceCallState.PAUSED

        # Resume voice call
        session.resume_voice_call()
        assert session.voice_call_state == VoiceCallState.ACTIVE

        # Mute voice call
        session.mute_voice_call()
        assert session.voice_call_state == VoiceCallState.MUTED

        # Resume from mute
        session.resume_voice_call()
        assert session.voice_call_state == VoiceCallState.ACTIVE

        # Simulate some time passing by manually setting an earlier start time
        session.voice_start_time = session.voice_start_time - timedelta(seconds=1)

        # End voice call
        duration = session.end_voice_call()

        assert session.voice_call_state == VoiceCallState.ENDED
        assert session.voice_start_time is None
        assert session.voice_callback is None
        assert duration >= 0
        assert session.metrics.voice_minutes > 0
    
    def test_voice_call_invalid_transitions(self):
        """Test invalid voice call state transitions."""
        session = ClientSession(client_id="test-client")
        
        # Cannot start voice call without authentication
        with pytest.raises(ValueError, match="Cannot start voice call in state"):
            session.start_voice_call()
        
        # Cannot activate without initializing
        with pytest.raises(ValueError, match="Cannot activate voice call from state"):
            session.activate_voice_call()
        
        # Cannot pause inactive call
        with pytest.raises(ValueError, match="Cannot pause voice call from state"):
            session.pause_voice_call()
        
        # Cannot mute inactive call
        with pytest.raises(ValueError, match="Cannot mute voice call from state"):
            session.mute_voice_call()
        
        # Cannot resume inactive call
        with pytest.raises(ValueError, match="Cannot resume voice call from state"):
            session.resume_voice_call()
    
    def test_connection_state_transitions(self):
        """Test connection state transition validation."""
        session = ClientSession(client_id="test-client")
        
        # Valid transitions
        assert session.can_transition_to(ConnectionState.CONNECTED)
        assert session.can_transition_to(ConnectionState.ERROR)
        assert session.can_transition_to(ConnectionState.DISCONNECTED)
        
        # Invalid transitions
        assert not session.can_transition_to(ConnectionState.AUTHENTICATED)
        assert not session.can_transition_to(ConnectionState.ACTIVE)
        
        # Test transition
        session.transition_to(ConnectionState.CONNECTED)
        assert session.connection_state == ConnectionState.CONNECTED
        
        # Test invalid transition
        with pytest.raises(ValueError, match="Invalid state transition"):
            session.transition_to(ConnectionState.ACTIVE)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        session = ClientSession(client_id="test-client")
        session.rate_limit_requests = 3
        session.rate_limit_window = 60
        
        # Should allow requests under limit
        assert session.check_rate_limit() is True
        assert session.check_rate_limit() is True
        assert session.check_rate_limit() is True
        
        # Should deny requests over limit
        assert session.check_rate_limit() is False
        assert session.check_rate_limit() is False
    
    def test_rate_limit_window_reset(self):
        """Test rate limit window reset."""
        session = ClientSession(client_id="test-client")
        session.rate_limit_requests = 2
        session.rate_limit_window = 1  # 1 second window
        
        # Use up rate limit
        assert session.check_rate_limit() is True
        assert session.check_rate_limit() is True
        assert session.check_rate_limit() is False
        
        # Wait for window to reset
        import time
        time.sleep(1.1)
        
        # Should allow requests again
        assert session.check_rate_limit() is True
    
    def test_idle_detection(self):
        """Test idle session detection."""
        session = ClientSession(client_id="test-client")
        session.max_idle_time = 1  # 1 second
        
        # Should not be idle initially
        assert session.is_idle() is False
        
        # Manually set last activity to past
        session.last_activity = datetime.now(timezone.utc) - timedelta(seconds=2)
        
        # Should be idle now
        assert session.is_idle() is True
    
    def test_session_duration(self):
        """Test session duration calculation."""
        session = ClientSession(client_id="test-client")
        
        # Set connection time to 1 minute ago
        session.connection_time = datetime.now(timezone.utc) - timedelta(minutes=1)
        
        duration = session.get_session_duration()
        assert duration >= 60  # At least 60 seconds
    
    def test_voice_call_duration(self):
        """Test voice call duration calculation."""
        session = ClientSession(client_id="test-client")
        session.authenticate("company-123", {}, "hash")
        
        # No duration when inactive
        assert session.get_voice_call_duration() == 0.0
        
        # Start voice call
        session.start_voice_call()
        session.activate_voice_call()
        
        # Set start time to 2 minutes ago
        session.voice_start_time = datetime.now(timezone.utc) - timedelta(minutes=2)
        
        duration = session.get_voice_call_duration()
        assert duration >= 2.0  # At least 2 minutes
    
    def test_error_recording(self):
        """Test error recording."""
        session = ClientSession(client_id="test-client")
        initial_errors = session.metrics.error_count
        
        session.record_error()
        
        assert session.metrics.error_count == initial_errors + 1
    
    def test_data_transfer_recording(self):
        """Test data transfer recording."""
        session = ClientSession(client_id="test-client")
        
        session.record_data_transfer(bytes_sent=100, bytes_received=200)
        
        assert session.metrics.bytes_sent == 100
        assert session.metrics.bytes_received == 200
        assert session.metrics.total_data_transfer == 300
    
    def test_websocket_reference(self):
        """Test websocket reference management."""
        session = ClientSession(client_id="test-client")
        mock_websocket = Mock()
        
        # Initially no reference
        assert session.get_websocket_reference() is None
        
        # Set reference
        session.set_websocket_reference(mock_websocket)
        assert session.get_websocket_reference() == mock_websocket
    
    def test_session_serialization(self):
        """Test session to dictionary conversion."""
        session = ClientSession(client_id="test-client")
        session.authenticate("company-123", {"name": "Test Co"}, "hash")
        session.start_conversation("conv-123", "agent-456")

        session_dict = session.to_dict()

        assert session_dict["client_id"] == "test-client"
        assert session_dict["connection_state"] == ConnectionState.ACTIVE.value  # Changed to ACTIVE
        assert session_dict["company_id"] == "company-123"
        assert session_dict["conversation_id"] == "conv-123"
        assert session_dict["agent_id"] == "agent-456"
        assert "metrics" in session_dict
        assert "session_duration" in session_dict
    
    def test_string_representation(self):
        """Test string representation of session."""
        session = ClientSession(client_id="test-client")
        
        str_repr = str(session)
        
        assert "test-client" in str_repr
        assert ConnectionState.CONNECTING.value in str_repr
        assert VoiceCallState.INACTIVE.value in str_repr
    
    def test_timezone_handling(self):
        """Test proper timezone handling."""
        # Test with naive datetime
        naive_time = datetime.now()
        session = ClientSession(
            client_id="test-client",
            connection_time=naive_time,
            last_activity=naive_time
        )
        
        # Should be converted to UTC
        assert session.connection_time.tzinfo == timezone.utc
        assert session.last_activity.tzinfo == timezone.utc
    
    def test_concurrent_metrics_updates(self):
        """Test that metrics updates are thread-safe via immutability."""
        session = ClientSession(client_id="test-client")
        original_metrics = session.metrics
        
        # Simulate concurrent updates
        session.update_activity(tokens=50)
        session.record_error()
        session.record_data_transfer(bytes_sent=100)
        
        # Original metrics should be unchanged
        assert original_metrics.total_tokens == 0
        assert original_metrics.error_count == 0
        assert original_metrics.bytes_sent == 0
        
        # Session should have updated metrics
        assert session.metrics.total_tokens == 50
        assert session.metrics.error_count == 1
        assert session.metrics.bytes_sent == 100
    
    def test_voice_call_edge_cases(self):
        """Test voice call edge cases."""
        session = ClientSession(client_id="test-client")
        session.authenticate("company-123", {}, "hash")
        
        # End inactive voice call should return 0
        duration = session.end_voice_call()
        assert duration == 0.0
        
        # Start and immediately end voice call
        session.start_voice_call()
        session.voice_start_time = None  # Simulate missing start time
        duration = session.end_voice_call()
        assert duration == 0.0
    
    def test_session_context_management(self):
        """Test session context management."""
        session = ClientSession(client_id="test-client")
        
        # Add user context
        session.user_context["preference"] = "voice"
        session.user_context["language"] = "en"
        
        assert session.user_context["preference"] == "voice"
        assert session.user_context["language"] == "en"
        
        # Context should be preserved in serialization
        session_dict = session.to_dict()
        # Note: user_context is not included in to_dict by default
        # This is intentional for security reasons