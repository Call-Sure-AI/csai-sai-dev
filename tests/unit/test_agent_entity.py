"""
Comprehensive tests for the Agent entity and related classes.

These tests cover all business logic, configuration management, metrics,
and edge cases for the agent domain entities.
"""

import pytest
from datetime import datetime, timezone, timedelta

from src.core.entities.agent import (
    Agent,
    AgentType,
    AgentCapability,
    AgentConfig,
    AgentMetrics
)


class TestAgentConfig:
    """Test the immutable AgentConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = AgentConfig(model_name="gpt-3.5-turbo")
        
        assert config.model_name == "gpt-3.5-turbo"
        assert config.model_version == "latest"
        assert config.provider == "openai"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.context_window_size == 4000
        assert config.function_calling_enabled is False
        assert config.streaming_enabled is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Empty model name
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            AgentConfig(model_name="")
        
        # Invalid temperature
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            AgentConfig(model_name="gpt-3.5-turbo", temperature=3.0)
        
        # Invalid max_tokens
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            AgentConfig(model_name="gpt-3.5-turbo", max_tokens=0)
        
        # Invalid top_p
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            AgentConfig(model_name="gpt-3.5-turbo", top_p=1.5)
        
        # Invalid context window size
        with pytest.raises(ValueError, match="context_window_size must be positive"):
            AgentConfig(model_name="gpt-3.5-turbo", context_window_size=-1)
    
    def test_config_immutability_methods(self):
        """Test immutable config update methods."""
        config = AgentConfig(model_name="gpt-3.5-turbo")
        
        # Test temperature update
        new_config = config.with_temperature(0.5)
        assert config.temperature == 0.7  # Original unchanged
        assert new_config.temperature == 0.5
        
        # Test max_tokens update
        new_config = config.with_max_tokens(2000)
        assert config.max_tokens == 1000  # Original unchanged
        assert new_config.max_tokens == 2000
        
        # Test functions update
        functions = ["get_weather", "send_email"]
        new_config = config.with_functions(functions)
        assert config.function_calling_enabled is False  # Original unchanged
        assert new_config.function_calling_enabled is True
        assert new_config.available_functions == functions
    
    def test_config_serialization(self):
        """Test config to dictionary conversion."""
        config = AgentConfig(
            model_name="gpt-4",
            temperature=0.8,
            max_tokens=1500,
            voice_model="tts-1",
            function_calling_enabled=True,
            available_functions=["test_function"]
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["model_name"] == "gpt-4"
        assert config_dict["temperature"] == 0.8
        assert config_dict["max_tokens"] == 1500
        assert config_dict["voice_model"] == "tts-1"
        assert config_dict["function_calling_enabled"] is True
        assert config_dict["available_functions"] == ["test_function"]


class TestAgentMetrics:
    """Test the immutable AgentMetrics class."""
    
    def test_default_metrics(self):
        """Test default metrics initialization."""
        metrics = AgentMetrics()
        
        assert metrics.total_conversations == 0
        assert metrics.total_messages == 0
        assert metrics.total_tokens_processed == 0
        assert metrics.total_tokens_generated == 0
        assert metrics.average_response_time == 0.0
        assert metrics.success_rate == 1.0
        assert metrics.error_count == 0
        assert metrics.function_success_rate == 1.0
        assert metrics.average_conversation_length == 0.0
    
    def test_add_conversation(self):
        """Test adding conversation to metrics."""
        metrics = AgentMetrics()
        new_metrics = metrics.add_conversation()
        
        assert metrics.total_conversations == 0  # Original unchanged
        assert new_metrics.total_conversations == 1
        assert new_metrics.last_active_time is not None
    
    def test_add_response_success(self):
        """Test adding successful response to metrics."""
        metrics = AgentMetrics()
        new_metrics = metrics.add_response(
            response_time=2.5,
            tokens_processed=100,
            tokens_generated=50,
            success=True
        )
        
        assert new_metrics.total_messages == 1
        assert new_metrics.total_tokens_processed == 100
        assert new_metrics.total_tokens_generated == 50
        assert new_metrics.average_response_time == 2.5
        assert new_metrics.average_tokens_per_response == 50.0
        assert new_metrics.success_rate == 1.0
        assert new_metrics.error_count == 0
    
    def test_add_response_failure(self):
        """Test adding failed response to metrics."""
        metrics = AgentMetrics()
        new_metrics = metrics.add_response(
            response_time=5.0,
            tokens_processed=100,
            tokens_generated=0,
            success=False
        )
        
        assert new_metrics.total_messages == 1
        assert new_metrics.error_count == 1
        assert new_metrics.success_rate == 0.0
    
    def test_multiple_responses_averaging(self):
        """Test averaging with multiple responses."""
        metrics = AgentMetrics()
        
        # First response: 2.0 seconds, 50 tokens
        metrics = metrics.add_response(2.0, 100, 50, True)
        # Second response: 4.0 seconds, 30 tokens
        metrics = metrics.add_response(4.0, 80, 30, True)
        
        assert metrics.total_messages == 2
        assert metrics.average_response_time == 3.0  # (2.0 + 4.0) / 2
        assert metrics.average_tokens_per_response == 40.0  # (50 + 30) / 2
        assert metrics.success_rate == 1.0
    
    def test_add_voice_session(self):
        """Test adding voice session to metrics."""
        metrics = AgentMetrics()
        new_metrics = metrics.add_voice_session(duration_minutes=5.5, quality_score=4.5)
        
        assert new_metrics.total_voice_minutes == 5.5
        assert new_metrics.voice_synthesis_count == 1
        assert new_metrics.average_voice_quality_score == 4.5
        
        # Add another session
        newer_metrics = new_metrics.add_voice_session(duration_minutes=3.0, quality_score=4.0)
        assert newer_metrics.total_voice_minutes == 8.5
        assert newer_metrics.voice_synthesis_count == 2
        assert newer_metrics.average_voice_quality_score == 4.25  # (4.5 + 4.0) / 2
    
    def test_add_function_call(self):
        """Test adding function call to metrics."""
        metrics = AgentMetrics()
        
        # Successful function call
        metrics = metrics.add_function_call(execution_time=1.5, success=True)
        assert metrics.function_calls_made == 1
        assert metrics.function_calls_successful == 1
        assert metrics.average_function_execution_time == 1.5
        assert metrics.function_success_rate == 1.0
        
        # Failed function call
        metrics = metrics.add_function_call(execution_time=3.0, success=False)
        assert metrics.function_calls_made == 2
        assert metrics.function_calls_successful == 1
        assert metrics.average_function_execution_time == 2.25  # (1.5 + 3.0) / 2
        assert metrics.function_success_rate == 0.5  # 1/2
    
    def test_add_feedback(self):
        """Test adding user feedback to metrics."""
        metrics = AgentMetrics()
        
        # First feedback
        metrics = metrics.add_feedback(4.5)
        assert metrics.feedback_count == 1
        assert metrics.average_satisfaction_score == 4.5
        
        # Second feedback
        metrics = metrics.add_feedback(3.5)
        assert metrics.feedback_count == 2
        assert metrics.average_satisfaction_score == 4.0  # (4.5 + 3.5) / 2
    
    def test_metrics_properties(self):
        """Test calculated properties of metrics."""
        metrics = AgentMetrics()
        
        # Test with no data
        assert metrics.function_success_rate == 1.0
        assert metrics.average_conversation_length == 0.0
        
        # Add some data
        metrics = metrics.add_conversation()
        metrics = metrics.add_response(2.0, 100, 50, True)
        metrics = metrics.add_response(3.0, 120, 60, True)
        
        assert metrics.average_conversation_length == 2.0  # 2 messages / 1 conversation


class TestAgent:
    """Test the Agent entity."""
    
    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        
        assert agent.id == "agent-123"
        assert agent.name == "Test Agent"
        assert agent.agent_type == AgentType.CONVERSATIONAL
        assert agent.is_active is True
        assert agent.is_available is True
        assert agent.current_load == 0
        assert isinstance(agent.config, AgentConfig)
        assert isinstance(agent.metrics, AgentMetrics)
    
    def test_agent_validation(self):
        """Test agent validation."""
        # Empty ID
        with pytest.raises(ValueError, match="id cannot be empty"):
            Agent(id="", name="Test Agent", agent_type=AgentType.CONVERSATIONAL)
        
        # Empty name
        with pytest.raises(ValueError, match="name cannot be empty"):
            Agent(id="agent-123", name="", agent_type=AgentType.CONVERSATIONAL)
    
    def test_default_capabilities_by_type(self):
        """Test default capabilities are set based on agent type."""
        # Conversational agent
        conv_agent = Agent(
            id="conv-1",
            name="Conv Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        assert AgentCapability.TEXT_GENERATION in conv_agent.capabilities
        assert AgentCapability.CONTEXT_MEMORY in conv_agent.capabilities
        assert AgentCapability.SENTIMENT_ANALYSIS in conv_agent.capabilities
        
        # Voice specialized agent
        voice_agent = Agent(
            id="voice-1",
            name="Voice Agent",
            agent_type=AgentType.VOICE_SPECIALIZED
        )
        assert AgentCapability.VOICE_SYNTHESIS in voice_agent.capabilities
        assert AgentCapability.VOICE_RECOGNITION in voice_agent.capabilities
        assert AgentCapability.REAL_TIME_PROCESSING in voice_agent.capabilities
        
        # Function calling agent
        func_agent = Agent(
            id="func-1",
            name="Function Agent",
            agent_type=AgentType.FUNCTION_CALLING
        )
        assert AgentCapability.FUNCTION_CALLING in func_agent.capabilities
        
        # Customer service agent
        cs_agent = Agent(
            id="cs-1",
            name="CS Agent",
            agent_type=AgentType.CUSTOMER_SERVICE
        )
        assert AgentCapability.SENTIMENT_ANALYSIS in cs_agent.capabilities
        assert AgentCapability.KNOWLEDGE_RETRIEVAL in cs_agent.capabilities
    
    def test_capability_management(self):
        """Test capability management."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        
        # Add capability
        agent.add_capability(AgentCapability.VOICE_SYNTHESIS)
        assert agent.has_capability(AgentCapability.VOICE_SYNTHESIS)
        
        # Remove capability
        agent.remove_capability(AgentCapability.VOICE_SYNTHESIS)
        assert not agent.has_capability(AgentCapability.VOICE_SYNTHESIS)
        
        # Check existing capability
        assert agent.has_capability(AgentCapability.TEXT_GENERATION)
    
    def test_config_management(self):
        """Test configuration management."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        
        initial_temp = agent.config.temperature
        
        # Update config
        agent.update_config(temperature=0.9, max_tokens=2000)
        
        assert agent.config.temperature == 0.9
        assert agent.config.max_tokens == 2000
        assert agent.config.model_name == "gpt-3.5-turbo"  # Unchanged
    
    def test_system_prompt_management(self):
        """Test system prompt management."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        
        new_prompt = "You are a specialized customer service agent."
        agent.set_system_prompt(new_prompt)
        
        assert agent.system_prompt == new_prompt
    
    def test_personality_traits(self):
        """Test personality trait management."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        
        # Add traits
        agent.add_personality_trait("friendliness", "high")
        agent.add_personality_trait("formality", "medium")
        
        assert agent.personality_traits["friendliness"] == "high"
        assert agent.personality_traits["formality"] == "medium"
        
        # Remove trait
        agent.remove_personality_trait("formality")
        assert "formality" not in agent.personality_traits
        assert "friendliness" in agent.personality_traits
    
    def test_conversation_management(self):
        """Test conversation load management."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        agent.max_concurrent_conversations = 2
        
        # Start conversations
        assert agent.can_handle_new_conversation() is True
        assert agent.start_conversation() is True
        assert agent.current_load == 1
        
        assert agent.start_conversation() is True
        assert agent.current_load == 2
        assert agent.is_overloaded() is True
        
        # Cannot start more conversations
        assert agent.can_handle_new_conversation() is False
        assert agent.start_conversation() is False
        assert agent.current_load == 2
        
        # End a conversation
        agent.end_conversation()
        assert agent.current_load == 1
        assert agent.can_handle_new_conversation() is True
    
    def test_agent_state_management(self):
        """Test agent activation and availability."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        
        # Initially active and available
        assert agent.is_active is True
        assert agent.is_available is True
        assert agent.can_handle_new_conversation() is True
        
        # Deactivate agent
        agent.deactivate()
        assert agent.is_active is False
        assert agent.can_handle_new_conversation() is False
        
        # Reactivate agent
        agent.activate()
        assert agent.is_active is True
        assert agent.can_handle_new_conversation() is True
        
        # Make unavailable
        agent.make_unavailable()
        assert agent.is_available is False
        assert agent.can_handle_new_conversation() is False
        
        # Make available again
        agent.make_available()
        assert agent.is_available is True
        assert agent.can_handle_new_conversation() is True
    
    def test_metrics_recording(self):
        """Test recording various metrics."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        
        # Record response
        agent.record_response(
            response_time=2.5,
            tokens_processed=100,
            tokens_generated=50,
            success=True
        )
        
        assert agent.metrics.total_messages == 1
        assert agent.metrics.average_response_time == 2.5
        assert agent.metrics.success_rate == 1.0
        
        # Record voice session
        agent.record_voice_session(duration_minutes=5.0, quality_score=4.5)
        assert agent.metrics.total_voice_minutes == 5.0
        assert agent.metrics.average_voice_quality_score == 4.5
        
        # Record function call
        agent.record_function_call(execution_time=1.5, success=True)
        assert agent.metrics.function_calls_made == 1
        assert agent.metrics.function_success_rate == 1.0
        
        # Record feedback
        agent.record_feedback(4.0)
        assert agent.metrics.average_satisfaction_score == 4.0
    
    def test_feedback_validation(self):
        """Test feedback score validation."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        
        # Valid scores
        agent.record_feedback(0.0)
        agent.record_feedback(5.0)
        agent.record_feedback(3.5)
        
        # Invalid scores
        with pytest.raises(ValueError, match="satisfaction_score must be between 0.0 and 5.0"):
            agent.record_feedback(-1.0)
        
        with pytest.raises(ValueError, match="satisfaction_score must be between 0.0 and 5.0"):
            agent.record_feedback(6.0)
    
    def test_load_percentage(self):
        """Test load percentage calculation."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        agent.max_concurrent_conversations = 4
        
        assert agent.get_load_percentage() == 0.0
        
        agent.start_conversation()
        assert agent.get_load_percentage() == 25.0
        
        agent.start_conversation()
        assert agent.get_load_percentage() == 50.0
        
        agent.start_conversation()
        agent.start_conversation()
        assert agent.get_load_percentage() == 100.0
    
    def test_load_percentage_edge_cases(self):
        """Test load percentage edge cases."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        agent.max_concurrent_conversations = 0
        
        # Should handle division by zero
        assert agent.get_load_percentage() == 0.0
    
    def test_uptime_calculation(self):
        """Test uptime calculation."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        
        # Set creation time to 1 hour ago
        agent.created_at = datetime.now(timezone.utc) - timedelta(hours=1)
        
        uptime = agent.get_uptime()
        assert uptime is not None
        assert uptime >= 3600  # At least 1 hour
        
        # Inactive agent should return None
        agent.deactivate()
        assert agent.get_uptime() is None
    
    def test_performance_score(self):
        """Test performance score calculation."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        
        # No messages should give perfect score
        assert agent.get_performance_score() == 100.0
        
        # Add some performance data
        agent.record_response(2.0, 100, 50, True)  # Fast, successful response
        agent.record_feedback(5.0)  # Perfect satisfaction
        
        score = agent.get_performance_score()
        assert 90.0 <= score <= 100.0  # Should be high
        
        # Add poor performance
        agent.record_response(25.0, 100, 50, False)  # Slow, failed response
        agent.record_feedback(1.0)  # Poor satisfaction
        
        score = agent.get_performance_score()
        assert score < 90.0  # Should be lower now
    
    def test_config_validation_method(self):
        """Test configuration validation method."""
        # Voice specialized agent without voice capabilities
        agent = Agent(
            id="agent-123",
            name="Voice Agent",
            agent_type=AgentType.VOICE_SPECIALIZED,
            capabilities=set()  # Empty capabilities
        )
        
        issues = agent.validate_config()
        assert len(issues) > 0
        assert any("Voice specialized agent missing capabilities" in issue for issue in issues)
        
        # Function calling agent without function capability
        func_agent = Agent(
            id="func-123",
            name="Function Agent",
            agent_type=AgentType.FUNCTION_CALLING,
            capabilities={AgentCapability.TEXT_GENERATION}  # Missing FUNCTION_CALLING
        )
        
        issues = func_agent.validate_config()
        assert any("Function calling agent missing FUNCTION_CALLING capability" in issue for issue in issues)
        
        # Agent with voice capability but no voice model
        voice_agent = Agent(
            id="voice-123",
            name="Voice Agent",
            agent_type=AgentType.CONVERSATIONAL,
            capabilities={AgentCapability.VOICE_SYNTHESIS}
        )
        
        issues = voice_agent.validate_config()
        assert any("Voice synthesis capability enabled but no voice model configured" in issue for issue in issues)
        
        # Agent with function calling enabled but no functions
        config = AgentConfig(model_name="gpt-3.5-turbo", function_calling_enabled=True)
        func_agent2 = Agent(
            id="func2-123",
            name="Function Agent 2",
            agent_type=AgentType.FUNCTION_CALLING,
            config=config
        )
        
        issues = func_agent2.validate_config()
        assert any("Function calling enabled but no functions configured" in issue for issue in issues)
        
        # Agent with invalid max conversations
        invalid_agent = Agent(
            id="invalid-123",
            name="Invalid Agent",
            agent_type=AgentType.CONVERSATIONAL,
            max_concurrent_conversations=0
        )
        
        issues = invalid_agent.validate_config()
        assert any("max_concurrent_conversations must be positive" in issue for issue in issues)
    
    def test_valid_configuration(self):
        """Test agent with valid configuration."""
        config = AgentConfig(
            model_name="gpt-4",
            voice_model="tts-1",
            function_calling_enabled=True,
            available_functions=["get_weather", "send_email"]
        )
        
        agent = Agent(
            id="valid-123",
            name="Valid Agent",
            agent_type=AgentType.VOICE_SPECIALIZED,
            config=config,
            capabilities={
                AgentCapability.TEXT_GENERATION,
                AgentCapability.VOICE_SYNTHESIS,
                AgentCapability.VOICE_RECOGNITION,
                AgentCapability.FUNCTION_CALLING
            }
        )
        
        issues = agent.validate_config()
        assert len(issues) == 0
    
    def test_agent_serialization(self):
        """Test agent to dictionary conversion."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL,
            description="A test agent for unit tests",
            version="2.0.0"
        )
        
        agent.add_personality_trait("friendliness", "high")
        agent.start_conversation()
        agent.record_response(2.0, 100, 50, True)
        
        agent_dict = agent.to_dict()
        
        assert agent_dict["id"] == "agent-123"
        assert agent_dict["name"] == "Test Agent"
        assert agent_dict["agent_type"] == AgentType.CONVERSATIONAL.value
        assert agent_dict["description"] == "A test agent for unit tests"
        assert agent_dict["version"] == "2.0.0"
        assert agent_dict["is_active"] is True
        assert agent_dict["is_available"] is True
        assert agent_dict["current_load"] == 1
        assert agent_dict["personality_traits"] == {"friendliness": "high"}
        assert "config" in agent_dict
        assert "metrics" in agent_dict
        assert "load_percentage" in agent_dict
        assert "performance_score" in agent_dict
        assert "can_handle_new_conversation" in agent_dict
    
    def test_string_representation(self):
        """Test string representation of agent."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        agent.start_conversation()
        agent.start_conversation()
        
        str_repr = str(agent)
        
        assert "agent-123" in str_repr
        assert "Test Agent" in str_repr
        assert AgentType.CONVERSATIONAL.value in str_repr
        assert "active" in str_repr
        assert "available" in str_repr
        assert "load=2" in str_repr
    
    def test_inactive_agent_string(self):
        """Test string representation of inactive agent."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        agent.deactivate()
        agent.make_unavailable()
        
        str_repr = str(agent)
        
        assert "inactive" in str_repr
        assert "unavailable" in str_repr
    
    def test_timezone_handling(self):
        """Test proper timezone handling."""
        # Test with naive datetime
        naive_time = datetime.now()
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL,
            created_at=naive_time,
            updated_at=naive_time
        )
        
        # Should be converted to UTC
        assert agent.created_at.tzinfo == timezone.utc
        assert agent.updated_at.tzinfo == timezone.utc
    
    def test_conversation_overflow_protection(self):
        """Test protection against conversation load overflow."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        agent.max_concurrent_conversations = 1
        agent.current_load = 5  # Manually set overflow
        
        # Should handle overflow gracefully
        agent.end_conversation()
        assert agent.current_load == 4  # Decremented but still above max
        
        # Should eventually reach normal levels
        for _ in range(5):
            agent.end_conversation()
        assert agent.current_load == 0
    
    def test_metrics_immutability(self):
        """Test that metrics objects are properly immutable."""
        agent = Agent(
            id="agent-123",
            name="Test Agent",
            agent_type=AgentType.CONVERSATIONAL
        )
        
        original_metrics = agent.metrics
        original_id = id(agent.metrics)
        
        # Record a response
        agent.record_response(2.0, 100, 50, True)
        
        # Metrics object should be different
        assert id(agent.metrics) != original_id
        assert original_metrics.total_messages == 0
        assert agent.metrics.total_messages == 1
    
    def test_agent_lifecycle(self):
        """Test complete agent lifecycle."""
        # Create agent with proper voice configuration
        config = AgentConfig(
            model_name="gpt-4",
            voice_model="tts-1"  # Add voice model to prevent validation error
        )
        
        agent = Agent(
            id="agent-123",
            name="Lifecycle Agent",
            agent_type=AgentType.VOICE_SPECIALIZED,
            config=config
        )
        
        # Initial state
        assert agent.is_active is True
        assert agent.can_handle_new_conversation() is True
        
        # Start handling conversations
        agent.start_conversation()
        agent.record_response(1.5, 100, 45, True)
        agent.record_voice_session(3.0, 4.5)
        agent.record_feedback(4.5)
        
        # Check performance
        assert agent.get_performance_score() > 80.0
        assert agent.metrics.total_conversations == 1
        
        # End conversation and check load
        agent.end_conversation()
        assert agent.current_load == 0
        
        # Deactivate for maintenance
        agent.deactivate()
        assert agent.can_handle_new_conversation() is False
        
        # Reactivate and continue
        agent.activate()
        assert agent.can_handle_new_conversation() is True
        
        # Validate final state
        issues = agent.validate_config()
        assert len(issues) == 0  # Should be valid