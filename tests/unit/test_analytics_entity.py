"""
Comprehensive tests for the Analytics entities.

These tests cover all business logic, calculations, and edge cases
for the analytics domain entities.
"""

import pytest
from datetime import datetime, timezone, timedelta

from src.core.entities.analytics import (
    SessionAnalytics,
    ConversationAnalytics,
    SystemMetrics,
    AnalyticsTimeframe,
    PerformanceMetric
)


class TestSessionAnalytics:
    """Test the SessionAnalytics entity."""
    
    def test_session_analytics_creation(self):
        """Test basic session analytics creation."""
        analytics = SessionAnalytics(
            session_id="session-123",
            client_id="client-456",
            company_id="company-789"
        )
        
        assert analytics.session_id == "session-123"
        assert analytics.client_id == "client-456"
        assert analytics.company_id == "company-789"
        assert analytics.duration_seconds == 0.0
        assert analytics.end_time is None
        assert analytics.is_completed is False
    
    def test_session_analytics_validation(self):
        """Test session analytics validation."""
        # Empty session_id
        with pytest.raises(ValueError, match="session_id cannot be empty"):
            SessionAnalytics(session_id="", client_id="client-456")
        
        # Empty client_id
        with pytest.raises(ValueError, match="client_id cannot be empty"):
            SessionAnalytics(session_id="session-123", client_id="")
    
    def test_create_from_session(self):
        """Test factory method for creating from session."""
        start_time = datetime.now(timezone.utc)
        analytics = SessionAnalytics.create_from_session(
            session_id="session-123",
            client_id="client-456",
            company_id="company-789",
            start_time=start_time
        )
        
        assert analytics.session_id == "session-123"
        assert analytics.start_time == start_time
    
    def test_with_completion(self):
        """Test session completion."""
        start_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        analytics = SessionAnalytics(
            session_id="session-123",
            client_id="client-456",
            start_time=start_time
        )
        
        end_time = datetime.now(timezone.utc)
        completed = analytics.with_completion(end_time)
        
        assert completed.end_time == end_time
        assert completed.duration_seconds >= 600  # At least 10 minutes
        assert completed.is_completed is True
        
        # Original should be unchanged
        assert analytics.end_time is None
        assert analytics.is_completed is False
    
    def test_with_message_stats(self):
        """Test message statistics update."""
        analytics = SessionAnalytics(
            session_id="session-123",
            client_id="client-456"
        )
        
        updated = analytics.with_message_stats(
            total_messages=10,
            user_messages=6,
            agent_messages=3,
            system_messages=1,
            error_messages=0
        )
        
        assert updated.total_messages == 10
        assert updated.user_messages == 6
        assert updated.agent_messages == 3
        assert updated.system_messages == 1
        assert updated.error_messages == 0
        assert updated.success_rate == 1.0  # No errors
        
        # Original unchanged
        assert analytics.total_messages == 0
    
    def test_with_voice_data(self):
        """Test voice data update."""
        analytics = SessionAnalytics(
            session_id="session-123",
            client_id="client-456"
        )
        
        updated = analytics.with_voice_data(
            duration_minutes=5.5,
            quality_score=4.2,
            interruptions=2
        )
        
        assert updated.voice_call_duration == 5.5
        assert updated.voice_quality_score == 4.2
        assert updated.audio_interruptions == 2
    
    def test_with_performance_data(self):
        """Test performance data update."""
        analytics = SessionAnalytics(
            session_id="session-123",
            client_id="client-456"
        )
        
        updated = analytics.with_performance_data(
            avg_response_time=2.5,
            max_response_time=5.0,
            min_response_time=1.0
        )
        
        assert updated.average_response_time == 2.5
        assert updated.max_response_time == 5.0
        assert updated.min_response_time == 1.0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        analytics = SessionAnalytics(
            session_id="session-123",
            client_id="client-456",
            total_messages=10,
            error_messages=2
        )
        
        assert analytics.success_rate == 0.8  # 8/10
        
        # No messages case
        empty_analytics = SessionAnalytics(
            session_id="session-123",
            client_id="client-456"
        )
        assert empty_analytics.success_rate == 1.0
    
    def test_function_success_rate(self):
        """Test function call success rate."""
        analytics = SessionAnalytics(
            session_id="session-123",
            client_id="client-456",
            function_calls_made=5,
            function_calls_successful=4
        )
        
        assert analytics.function_success_rate == 0.8  # 4/5
        
        # No function calls case
        no_functions = SessionAnalytics(
            session_id="session-123",
            client_id="client-456"
        )
        assert no_functions.function_success_rate == 1.0
    
    def test_total_data_transfer(self):
        """Test total data transfer calculation."""
        analytics = SessionAnalytics(
            session_id="session-123",
            client_id="client-456",
            bytes_sent=1000,
            bytes_received=2000
        )
        
        assert analytics.total_data_transfer == 3000
    
    def test_timezone_handling(self):
        """Test timezone handling in session analytics."""
        naive_time = datetime.now()
        analytics = SessionAnalytics(
            session_id="session-123",
            client_id="client-456",
            start_time=naive_time
        )
        
        assert analytics.start_time.tzinfo == timezone.utc
    
    def test_session_analytics_serialization(self):
        """Test session analytics serialization."""
        analytics = SessionAnalytics(
            session_id="session-123",
            client_id="client-456",
            company_id="company-789",
            total_messages=5,
            user_messages=3,
            agent_messages=2,
            voice_call_duration=3.5,
            user_satisfaction_score=4.5
        )
        
        data = analytics.to_dict()
        
        assert data["session_id"] == "session-123"
        assert data["client_id"] == "client-456"
        assert data["company_id"] == "company-789"
        assert data["total_messages"] == 5
        assert data["voice_call_duration"] == 3.5
        assert data["user_satisfaction_score"] == 4.5
        assert data["is_completed"] is False
        assert "start_time" in data
        assert "success_rate" in data
        assert "total_data_transfer" in data


class TestConversationAnalytics:
    """Test the ConversationAnalytics entity."""
    
    def test_conversation_analytics_creation(self):
        """Test basic conversation analytics creation."""
        analytics = ConversationAnalytics(
            conversation_id="conv-123",
            client_id="client-456",
            agent_id="agent-789",
            company_id="company-000"
        )
        
        assert analytics.conversation_id == "conv-123"
        assert analytics.client_id == "client-456"
        assert analytics.agent_id == "agent-789"
        assert analytics.company_id == "company-000"
        assert analytics.duration_seconds == 0.0
        assert analytics.is_completed is False
    
    def test_conversation_analytics_validation(self):
        """Test conversation analytics validation."""
        # Empty conversation_id
        with pytest.raises(ValueError, match="conversation_id cannot be empty"):
            ConversationAnalytics(conversation_id="", client_id="client-456")
        
        # Empty client_id
        with pytest.raises(ValueError, match="client_id cannot be empty"):
            ConversationAnalytics(conversation_id="conv-123", client_id="")
    
    def test_create_from_conversation(self):
        """Test factory method for creating from conversation."""
        analytics = ConversationAnalytics.create_from_conversation(
            conversation_id="conv-123",
            client_id="client-456",
            agent_id="agent-789",
            company_id="company-000"
        )
        
        assert analytics.conversation_id == "conv-123"
        assert analytics.agent_id == "agent-789"
    
    def test_with_completion(self):
        """Test conversation completion."""
        start_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        analytics = ConversationAnalytics(
            conversation_id="conv-123",
            client_id="client-456",
            start_time=start_time
        )
        
        end_time = datetime.now(timezone.utc)
        completed = analytics.with_completion(
            end_time=end_time,
            completion_reason="user_ended",
            goal_achieved=True
        )
        
        assert completed.end_time == end_time
        assert completed.duration_seconds >= 300  # At least 5 minutes
        assert completed.completion_reason == "user_ended"
        assert completed.goal_achieved is True
        assert completed.is_completed is True
        
        # Original unchanged
        assert analytics.is_completed is False
    
    def test_with_response_time(self):
        """Test response time tracking."""
        analytics = ConversationAnalytics(
            conversation_id="conv-123",
            client_id="client-456"
        )
        
        # Add first response time
        updated1 = analytics.with_response_time(2.0)
        assert updated1.average_response_time == 2.0
        assert updated1.median_response_time == 2.0
        assert 2.0 in updated1.response_times
        
        # Add second response time
        updated2 = updated1.with_response_time(4.0)
        assert updated2.average_response_time == 3.0  # (2.0 + 4.0) / 2
        assert len(updated2.response_times) == 2
        
        # Add third response time to test median calculation
        updated3 = updated2.with_response_time(6.0)
        assert updated3.average_response_time == 4.0  # (2.0 + 4.0 + 6.0) / 3
        assert updated3.median_response_time == 4.0  # Middle value of [2.0, 4.0, 6.0]
        
        # Original unchanged
        assert analytics.average_response_time == 0.0
    
    def test_function_success_rate(self):
        """Test function call success rate."""
        analytics = ConversationAnalytics(
            conversation_id="conv-123",
            client_id="client-456",
            function_call_count=4,
            function_success_count=3
        )
        
        assert analytics.function_success_rate == 0.75  # 3/4
        
        # No function calls case
        no_functions = ConversationAnalytics(
            conversation_id="conv-123",
            client_id="client-456"
        )
        assert no_functions.function_success_rate == 1.0
    
    def test_messages_per_minute(self):
        """Test messages per minute calculation."""
        analytics = ConversationAnalytics(
            conversation_id="conv-123",
            client_id="client-456",
            message_count=10,
            duration_seconds=120  # 2 minutes
        )
        
        assert analytics.messages_per_minute == 5.0  # 10 messages / 2 minutes
        
        # Zero duration case
        zero_duration = ConversationAnalytics(
            conversation_id="conv-123",
            client_id="client-456",
            message_count=10,
            duration_seconds=0
        )
        assert zero_duration.messages_per_minute == 0.0
    
    def test_tokens_per_minute(self):
        """Test tokens per minute calculation."""
        analytics = ConversationAnalytics(
            conversation_id="conv-123",
            client_id="client-456",
            total_tokens=600,
            duration_seconds=120  # 2 minutes
        )
        
        assert analytics.tokens_per_minute == 300.0  # 600 tokens / 2 minutes
        
        # Zero duration case
        zero_duration = ConversationAnalytics(
            conversation_id="conv-123",
            client_id="client-456",
            total_tokens=600,
            duration_seconds=0
        )
        assert zero_duration.tokens_per_minute == 0.0
    
    def test_response_time_percentiles(self):
        """Test response time percentile calculations."""
        analytics = ConversationAnalytics(
            conversation_id="conv-123",
            client_id="client-456"
        )
        
        # Add multiple response times
        response_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        for time in response_times:
            analytics = analytics.with_response_time(time)
        
        # Check calculations
        assert analytics.average_response_time == 5.5  # Average of 1-10
        assert analytics.median_response_time == 5.5  # Median of 1-10
        assert analytics.response_time_p95 >= 9.0  # 95th percentile should be high


class TestSystemMetrics:
    """Test the SystemMetrics entity."""
    
    def test_system_metrics_creation(self):
        """Test basic system metrics creation."""
        metrics = SystemMetrics(
            timeframe=AnalyticsTimeframe.HOURLY,
            active_connections=50,
            active_agents=5
        )
        
        assert metrics.timeframe == AnalyticsTimeframe.HOURLY
        assert metrics.active_connections == 50
        assert metrics.active_agents == 5
        assert metrics.timestamp.tzinfo == timezone.utc
    
    def test_create_snapshot(self):
        """Test system metrics snapshot creation."""
        snapshot = SystemMetrics.create_snapshot(AnalyticsTimeframe.DAILY)
        
        assert snapshot.timeframe == AnalyticsTimeframe.DAILY
        assert isinstance(snapshot.timestamp, datetime)
    
    def test_with_connections(self):
        """Test connection metrics update."""
        metrics = SystemMetrics()
        
        updated = metrics.with_connections(
            active=100,
            total_today=500,
            peak_concurrent=150,
            success_rate=0.98
        )
        
        assert updated.active_connections == 100
        assert updated.total_connections_today == 500
        assert updated.peak_concurrent_connections == 150
        assert updated.connection_success_rate == 0.98
        
        # Original unchanged
        assert metrics.active_connections == 0
    
    def test_with_performance(self):
        """Test performance metrics update."""
        metrics = SystemMetrics()
        
        updated = metrics.with_performance(
            avg_response=2.5,
            median_response=2.0,
            p95_response=5.0,
            p99_response=8.0
        )
        
        assert updated.average_response_time == 2.5
        assert updated.median_response_time == 2.0
        assert updated.p95_response_time == 5.0
        assert updated.p99_response_time == 8.0
    
    def test_with_resource_usage(self):
        """Test resource usage metrics update."""
        metrics = SystemMetrics()
        
        updated = metrics.with_resource_usage(
            cpu_percent=75.0,
            memory_percent=80.0,
            disk_percent=60.0,
            network_mbps=100.0
        )
        
        assert updated.cpu_usage_percent == 75.0
        assert updated.memory_usage_percent == 80.0
        assert updated.disk_usage_percent == 60.0
        assert updated.network_usage_mbps == 100.0

    def test_system_health_score(self):
        """Test system health score calculation."""
        # Good performance metrics
        good_metrics = SystemMetrics(
            average_response_time=1.0,  # Fast response
            error_rate=0.01,  # 1% error rate
            cpu_usage_percent=40.0,  # Lower CPU usage
            memory_usage_percent=50.0,  # Lower memory usage
            disk_usage_percent=30.0,  # Low disk usage
            average_satisfaction_score=4.8  # Higher satisfaction
        )
        
        health_score = good_metrics.system_health_score
        assert 80.0 <= health_score <= 100.0  # Should be high
        
        # Poor performance metrics
        poor_metrics = SystemMetrics(
            average_response_time=10.0,  # Slow response
            error_rate=0.1,  # 10% error rate
            cpu_usage_percent=95.0,  # High CPU usage
            memory_usage_percent=98.0,  # High memory usage
            disk_usage_percent=90.0,  # High disk usage
            average_satisfaction_score=2.0  # Low satisfaction
        )
        
        poor_health_score = poor_metrics.system_health_score
        assert poor_health_score < 50.0  # Should be low
    
    def test_is_overloaded(self):
        """Test overload detection."""
        # Normal system
        normal_metrics = SystemMetrics(
            cpu_usage_percent=70.0,
            memory_usage_percent=75.0,
            error_rate=0.02,
            average_response_time=2.0
        )
        assert normal_metrics.is_overloaded is False
        
        # High CPU
        high_cpu = SystemMetrics(cpu_usage_percent=95.0)
        assert high_cpu.is_overloaded is True
        
        # High memory
        high_memory = SystemMetrics(memory_usage_percent=95.0)
        assert high_memory.is_overloaded is True
        
        # High error rate
        high_errors = SystemMetrics(error_rate=0.06)  # 6%
        assert high_errors.is_overloaded is True
        
        # Slow response time
        slow_response = SystemMetrics(average_response_time=15.0)
        assert slow_response.is_overloaded is True
    
    def test_system_metrics_serialization(self):
        """Test system metrics serialization."""
        metrics = SystemMetrics(
            timeframe=AnalyticsTimeframe.DAILY,
            active_connections=100,
            active_agents=10,
            average_response_time=2.5,
            error_rate=0.02,
            cpu_usage_percent=75.0,
            requests_by_region={"us-east": 100, "eu-west": 50},
            api_calls_per_endpoint={"/chat": 200, "/voice": 150}
        )
        
        data = metrics.to_dict()
        
        assert data["timeframe"] == AnalyticsTimeframe.DAILY.value
        assert data["active_connections"] == 100
        assert data["active_agents"] == 10
        assert data["average_response_time"] == 2.5
        assert data["error_rate"] == 0.02
        assert data["cpu_usage_percent"] == 75.0
        assert data["requests_by_region"] == {"us-east": 100, "eu-west": 50}
        assert data["api_calls_per_endpoint"] == {"/chat": 200, "/voice": 150}
        assert "system_health_score" in data
        assert "is_overloaded" in data
        assert "timestamp" in data
    
    def test_timezone_handling_system_metrics(self):
        """Test timezone handling in system metrics."""
        naive_time = datetime.now()
        metrics = SystemMetrics(timestamp=naive_time)
        
        assert metrics.timestamp.tzinfo == timezone.utc
    
    def test_edge_case_calculations(self):
        """Test edge cases in metric calculations."""
        # Zero values
        zero_metrics = SystemMetrics(
            average_response_time=0.0,
            error_rate=0.0,
            cpu_usage_percent=0.0
        )
        
        # Should handle gracefully
        health_score = zero_metrics.system_health_score
        assert 0.0 <= health_score <= 100.0
        
        # Maximum values
        max_metrics = SystemMetrics(
            cpu_usage_percent=100.0,
            memory_usage_percent=100.0,
            error_rate=1.0,  # 100% error rate
            average_response_time=1000.0
        )
        
        max_health_score = max_metrics.system_health_score
        assert max_health_score >= 0.0  # Should not be negative
        assert max_metrics.is_overloaded is True


class TestAnalyticsEnums:
    """Test analytics-related enums."""
    
    def test_analytics_timeframe_values(self):
        """Test AnalyticsTimeframe enum values."""
        assert AnalyticsTimeframe.REAL_TIME.value == "real_time"
        assert AnalyticsTimeframe.HOURLY.value == "hourly"
        assert AnalyticsTimeframe.DAILY.value == "daily"
        assert AnalyticsTimeframe.WEEKLY.value == "weekly"
        assert AnalyticsTimeframe.MONTHLY.value == "monthly"
        assert AnalyticsTimeframe.YEARLY.value == "yearly"
    
    def test_performance_metric_values(self):
        """Test PerformanceMetric enum values."""
        assert PerformanceMetric.RESPONSE_TIME.value == "response_time"
        assert PerformanceMetric.TOKEN_USAGE.value == "token_usage"
        assert PerformanceMetric.SUCCESS_RATE.value == "success_rate"
        assert PerformanceMetric.ERROR_RATE.value == "error_rate"
        assert PerformanceMetric.VOICE_QUALITY.value == "voice_quality"
        assert PerformanceMetric.USER_SATISFACTION.value == "user_satisfaction"
        assert PerformanceMetric.FUNCTION_CALL_SUCCESS.value == "function_call_success"
        assert PerformanceMetric.THROUGHPUT.value == "throughput"
        assert PerformanceMetric.LATENCY.value == "latency"
        assert PerformanceMetric.AVAILABILITY.value == "availability"
        assert PerformanceMetric.RESOURCE_USAGE.value == "resource_usage"


class TestAnalyticsIntegration:
    """Test integration scenarios between analytics entities."""
    
    def test_session_to_conversation_analytics(self):
        """Test deriving conversation analytics from session analytics."""
        session = SessionAnalytics(
            session_id="session-123",
            client_id="client-456",
            company_id="company-789",
            total_messages=10,
            user_messages=6,
            agent_messages=4,
            total_tokens=500,
            voice_call_duration=5.0,
            user_satisfaction_score=4.5
        )
        
        # Create conversation analytics with similar data
        conversation = ConversationAnalytics.create_from_conversation(
            conversation_id="conv-123",
            client_id=session.client_id,
            company_id=session.company_id
        )
        
        # Verify related fields can be derived
        assert conversation.client_id == session.client_id
        assert conversation.company_id == session.company_id
    
    def test_analytics_immutability(self):
        """Test that all analytics objects are properly immutable."""
        session = SessionAnalytics(
            session_id="session-123",
            client_id="client-456"
        )
        original_id = id(session)
        
        # Create updated version
        updated = session.with_message_stats(5, 3, 2)
        
        # Should be different objects
        assert id(updated) != original_id
        assert session.total_messages == 0
        assert updated.total_messages == 5
        
        # Same test for conversation analytics
        conv = ConversationAnalytics(
            conversation_id="conv-123",
            client_id="client-456"
        )
        conv_original_id = id(conv)
        
        conv_updated = conv.with_response_time(2.5)
        
        assert id(conv_updated) != conv_original_id
        assert conv.average_response_time == 0.0
        assert conv_updated.average_response_time == 2.5
    
    def test_comprehensive_analytics_workflow(self):
        """Test a complete analytics workflow."""
        # Start session
        session = SessionAnalytics.create_from_session(
            session_id="session-123",
            client_id="client-456",
            company_id="company-789"
        )
        
        # Add message activity
        session = session.with_message_stats(
            total_messages=8,
            user_messages=5,
            agent_messages=3
        )
        
        # Add voice activity
        session = session.with_voice_data(
            duration_minutes=4.5,
            quality_score=4.2,
            interruptions=1
        )
        
        # Add performance data
        session = session.with_performance_data(
            avg_response_time=2.1,
            max_response_time=4.0,
            min_response_time=1.0
        )
        
        # Complete session
        session = session.with_completion()
        
        # Verify final state
        assert session.is_completed is True
        assert session.total_messages == 8
        assert session.voice_call_duration == 4.5
        assert session.average_response_time == 2.1
        assert session.success_rate == 1.0  # No errors
        
        # Create system metrics snapshot
        system = SystemMetrics.create_snapshot(AnalyticsTimeframe.REAL_TIME)
        system = system.with_connections(
            active=1,  # Our session
            total_today=10,
            peak_concurrent=5,
            success_rate=1.0
        )
        
        system = system.with_performance(
            avg_response=session.average_response_time,
            median_response=2.0,
            p95_response=3.5,
            p99_response=4.0
        )
        
        # Verify system metrics reflect session data
        assert system.average_response_time == session.average_response_time
        assert system.active_connections == 1