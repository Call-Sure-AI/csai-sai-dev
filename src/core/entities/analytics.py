"""
Analytics domain entities for the AI voice calling system.

This module contains entities related to analytics, metrics collection,
and performance tracking across sessions, conversations, and system-wide metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from uuid import uuid4


class AnalyticsTimeframe(Enum):
    """Time frames for analytics aggregation."""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class PerformanceMetric(Enum):
    """Types of performance metrics."""
    RESPONSE_TIME = "response_time"
    TOKEN_USAGE = "token_usage"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    VOICE_QUALITY = "voice_quality"
    USER_SATISFACTION = "user_satisfaction"
    FUNCTION_CALL_SUCCESS = "function_call_success"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    AVAILABILITY = "availability"
    RESOURCE_USAGE = "resource_usage"


@dataclass(frozen=True)
class SessionAnalytics:
    """
    Immutable analytics data for a client session.
    """
    session_id: str
    client_id: str
    company_id: Optional[str] = None
    
    # Session metadata
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Communication metrics
    total_messages: int = 0
    user_messages: int = 0
    agent_messages: int = 0
    system_messages: int = 0
    error_messages: int = 0
    
    # Token usage
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Voice metrics
    voice_call_duration: float = 0.0  # minutes
    voice_quality_score: Optional[float] = None
    audio_interruptions: int = 0
    
    # Performance metrics
    average_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = 0.0
    
    # Error tracking
    error_count: int = 0
    timeout_count: int = 0
    connection_drops: int = 0
    
    # Data transfer
    bytes_sent: int = 0
    bytes_received: int = 0
    
    # User experience
    user_satisfaction_score: Optional[float] = None
    completion_rate: float = 1.0  # 0.0 to 1.0
    
    # Function calls
    function_calls_made: int = 0
    function_calls_successful: int = 0
    
    # Geographic and technical data
    user_location: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.session_id:
            raise ValueError("session_id cannot be empty")
        if not self.client_id:
            raise ValueError("client_id cannot be empty")
        
        # Ensure timestamps are timezone-aware
        if self.start_time.tzinfo is None:
            object.__setattr__(self, 'start_time', self.start_time.replace(tzinfo=timezone.utc))
        if self.end_time and self.end_time.tzinfo is None:
            object.__setattr__(self, 'end_time', self.end_time.replace(tzinfo=timezone.utc))
    
    @classmethod
    def create_from_session(
        cls,
        session_id: str,
        client_id: str,
        company_id: Optional[str] = None,
        start_time: Optional[datetime] = None
    ) -> 'SessionAnalytics':
        """Create session analytics from basic session data."""
        return cls(
            session_id=session_id,
            client_id=client_id,
            company_id=company_id,
            start_time=start_time or datetime.now(timezone.utc)
        )
    
    def with_completion(self, end_time: Optional[datetime] = None) -> 'SessionAnalytics':
        """Create new analytics with session completion data."""
        end_time = end_time or datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds()
        
        return SessionAnalytics(
            **{
                **self.__dict__,
                'end_time': end_time,
                'duration_seconds': duration
            }
        )
    
    def with_message_stats(
        self,
        total_messages: int,
        user_messages: int,
        agent_messages: int,
        system_messages: int = 0,
        error_messages: int = 0
    ) -> 'SessionAnalytics':
        """Create new analytics with message statistics."""
        return SessionAnalytics(
            **{
                **self.__dict__,
                'total_messages': total_messages,
                'user_messages': user_messages,
                'agent_messages': agent_messages,
                'system_messages': system_messages,
                'error_messages': error_messages
            }
        )
    
    def with_voice_data(
        self,
        duration_minutes: float,
        quality_score: Optional[float] = None,
        interruptions: int = 0
    ) -> 'SessionAnalytics':
        """Create new analytics with voice call data."""
        return SessionAnalytics(
            **{
                **self.__dict__,
                'voice_call_duration': duration_minutes,
                'voice_quality_score': quality_score,
                'audio_interruptions': interruptions
            }
        )
    
    def with_performance_data(
        self,
        avg_response_time: float,
        max_response_time: float,
        min_response_time: float
    ) -> 'SessionAnalytics':
        """Create new analytics with performance data."""
        return SessionAnalytics(
            **{
                **self.__dict__,
                'average_response_time': avg_response_time,
                'max_response_time': max_response_time,
                'min_response_time': min_response_time
            }
        )
    
    @property
    def is_completed(self) -> bool:
        """Check if session is completed."""
        return self.end_time is not None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate based on errors vs total messages."""
        if self.total_messages == 0:
            return 1.0
        return (self.total_messages - self.error_messages) / self.total_messages
    
    @property
    def function_success_rate(self) -> float:
        """Calculate function call success rate."""
        if self.function_calls_made == 0:
            return 1.0
        return self.function_calls_successful / self.function_calls_made
    
    @property
    def total_data_transfer(self) -> int:
        """Calculate total data transfer in bytes."""
        return self.bytes_sent + self.bytes_received
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "client_id": self.client_id,
            "company_id": self.company_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "total_messages": self.total_messages,
            "user_messages": self.user_messages,
            "agent_messages": self.agent_messages,
            "system_messages": self.system_messages,
            "error_messages": self.error_messages,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "voice_call_duration": self.voice_call_duration,
            "voice_quality_score": self.voice_quality_score,
            "audio_interruptions": self.audio_interruptions,
            "average_response_time": self.average_response_time,
            "max_response_time": self.max_response_time,
            "min_response_time": self.min_response_time,
            "error_count": self.error_count,
            "timeout_count": self.timeout_count,
            "connection_drops": self.connection_drops,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "total_data_transfer": self.total_data_transfer,
            "user_satisfaction_score": self.user_satisfaction_score,
            "completion_rate": self.completion_rate,
            "success_rate": self.success_rate,
            "function_calls_made": self.function_calls_made,
            "function_calls_successful": self.function_calls_successful,
            "function_success_rate": self.function_success_rate,
            "user_location": self.user_location,
            "user_agent": self.user_agent,
            "ip_address": self.ip_address,
            "is_completed": self.is_completed
        }


@dataclass(frozen=True)
class ConversationAnalytics:
    """
    Immutable analytics data for a conversation.
    """
    conversation_id: str
    client_id: str
    agent_id: Optional[str] = None
    company_id: Optional[str] = None
    
    # Conversation metadata
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Message flow
    message_count: int = 0
    turns_count: int = 0  # Back-and-forth exchanges
    average_user_message_length: float = 0.0
    average_agent_message_length: float = 0.0
    
    # Response metrics
    response_times: List[float] = field(default_factory=list)
    average_response_time: float = 0.0
    median_response_time: float = 0.0
    response_time_p95: float = 0.0
    
    # Token economics
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_estimate: float = 0.0
    
    # Quality metrics
    conversation_quality_score: Optional[float] = None
    sentiment_scores: List[float] = field(default_factory=list)
    average_sentiment: Optional[float] = None
    
    # Function calling
    functions_used: List[str] = field(default_factory=list)
    function_call_count: int = 0
    function_success_count: int = 0
    
    # Voice specific
    voice_segments: int = 0
    total_voice_duration: float = 0.0
    voice_interruptions: int = 0
    speech_to_text_accuracy: Optional[float] = None
    
    # Outcome
    goal_achieved: Optional[bool] = None
    completion_reason: Optional[str] = None
    user_satisfaction: Optional[float] = None
    
    # Context
    topics_discussed: List[str] = field(default_factory=list)
    context_switches: int = 0
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.conversation_id:
            raise ValueError("conversation_id cannot be empty")
        if not self.client_id:
            raise ValueError("client_id cannot be empty")
        
        # Ensure timestamps are timezone-aware
        if self.start_time.tzinfo is None:
            object.__setattr__(self, 'start_time', self.start_time.replace(tzinfo=timezone.utc))
        if self.end_time and self.end_time.tzinfo is None:
            object.__setattr__(self, 'end_time', self.end_time.replace(tzinfo=timezone.utc))
    
    @classmethod
    def create_from_conversation(
        cls,
        conversation_id: str,
        client_id: str,
        agent_id: Optional[str] = None,
        company_id: Optional[str] = None
    ) -> 'ConversationAnalytics':
        """Create conversation analytics from basic conversation data."""
        return cls(
            conversation_id=conversation_id,
            client_id=client_id,
            agent_id=agent_id,
            company_id=company_id
        )
    
    def with_completion(
        self,
        end_time: Optional[datetime] = None,
        completion_reason: Optional[str] = None,
        goal_achieved: Optional[bool] = None
    ) -> 'ConversationAnalytics':
        """Create new analytics with conversation completion data."""
        end_time = end_time or datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds()
        
        return ConversationAnalytics(
            **{
                **self.__dict__,
                'end_time': end_time,
                'duration_seconds': duration,
                'completion_reason': completion_reason,
                'goal_achieved': goal_achieved
            }
        )
    
    def with_response_time(self, response_time: float) -> 'ConversationAnalytics':
        """Create new analytics with added response time."""
        new_times = self.response_times + [response_time]
        new_avg = sum(new_times) / len(new_times)
        
        # Calculate median and p95
        sorted_times = sorted(new_times)
        n = len(sorted_times)
        
        # Proper median calculation
        if n % 2 == 0:
            new_median = (sorted_times[n//2 - 1] + sorted_times[n//2]) / 2.0
        else:
            new_median = sorted_times[n//2]
        
        # P95 calculation
        p95_idx = min(int(n * 0.95), n - 1)
        new_p95 = sorted_times[p95_idx] if sorted_times else 0.0
        
        return ConversationAnalytics(
            **{
                **self.__dict__,
                'response_times': new_times,
                'average_response_time': new_avg,
                'median_response_time': new_median,
                'response_time_p95': new_p95
            }
        )
    
    @property
    def is_completed(self) -> bool:
        """Check if conversation is completed."""
        return self.end_time is not None
    
    @property
    def function_success_rate(self) -> float:
        """Calculate function call success rate."""
        if self.function_call_count == 0:
            return 1.0
        return self.function_success_count / self.function_call_count
    
    @property
    def messages_per_minute(self) -> float:
        """Calculate messages per minute."""
        if self.duration_seconds == 0:
            return 0.0
        return (self.message_count / self.duration_seconds) * 60
    
    @property
    def tokens_per_minute(self) -> float:
        """Calculate tokens per minute."""
        if self.duration_seconds == 0:
            return 0.0
        return (self.total_tokens / self.duration_seconds) * 60


@dataclass(frozen=True)
class SystemMetrics:
    """
    Immutable system-wide metrics and performance data.
    """
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeframe: AnalyticsTimeframe = AnalyticsTimeframe.REAL_TIME
    
    # Connection metrics
    active_connections: int = 0
    total_connections_today: int = 0
    peak_concurrent_connections: int = 0
    connection_success_rate: float = 1.0
    
    # Agent metrics
    active_agents: int = 0
    total_agents: int = 0
    average_agent_load: float = 0.0
    overloaded_agents: int = 0
    
    # Performance metrics
    average_response_time: float = 0.0
    median_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # Throughput
    requests_per_second: float = 0.0
    messages_per_second: float = 0.0
    tokens_per_second: float = 0.0
    
    # Error rates
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    connection_drop_rate: float = 0.0
    
    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_usage_mbps: float = 0.0
    
    # Voice metrics
    active_voice_calls: int = 0
    voice_quality_score: float = 0.0
    voice_latency_ms: float = 0.0
    
    # Cost metrics
    total_tokens_today: int = 0
    estimated_cost_today: float = 0.0
    cost_per_conversation: float = 0.0
    
    # User satisfaction
    average_satisfaction_score: Optional[float] = None
    satisfaction_responses_count: int = 0
    
    # Geographic distribution
    requests_by_region: Dict[str, int] = field(default_factory=dict)
    
    # API usage
    api_calls_per_endpoint: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.timestamp.tzinfo is None:
            object.__setattr__(self, 'timestamp', self.timestamp.replace(tzinfo=timezone.utc))
    
    @classmethod
    def create_snapshot(
        cls,
        timeframe: AnalyticsTimeframe = AnalyticsTimeframe.REAL_TIME
    ) -> 'SystemMetrics':
        """Create a system metrics snapshot."""
        return cls(timeframe=timeframe)
    
    def with_connections(
        self,
        active: int,
        total_today: int,
        peak_concurrent: int,
        success_rate: float
    ) -> 'SystemMetrics':
        """Create new metrics with connection data."""
        return SystemMetrics(
            **{
                **self.__dict__,
                'active_connections': active,
                'total_connections_today': total_today,
                'peak_concurrent_connections': peak_concurrent,
                'connection_success_rate': success_rate
            }
        )
    
    def with_performance(
        self,
        avg_response: float,
        median_response: float,
        p95_response: float,
        p99_response: float
    ) -> 'SystemMetrics':
        """Create new metrics with performance data."""
        return SystemMetrics(
            **{
                **self.__dict__,
                'average_response_time': avg_response,
                'median_response_time': median_response,
                'p95_response_time': p95_response,
                'p99_response_time': p99_response
            }
        )
    
    def with_resource_usage(
        self,
        cpu_percent: float,
        memory_percent: float,
        disk_percent: float,
        network_mbps: float
    ) -> 'SystemMetrics':
        """Create new metrics with resource usage data."""
        return SystemMetrics(
            **{
                **self.__dict__,
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory_percent,
                'disk_usage_percent': disk_percent,
                'network_usage_mbps': network_mbps
            }
        )
    
    @property
    def system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        # Weighted scoring of various metrics
        performance_weight = 0.3
        error_weight = 0.3
        resource_weight = 0.2
        satisfaction_weight = 0.2
        
        # Performance score (inverse of response time)
        max_acceptable_response = 5.0  # seconds
        performance_score = max(
            0,
            (max_acceptable_response - self.average_response_time) / max_acceptable_response * 100
        )
        
        # Error score (inverse of error rate)
        error_score = (1.0 - self.error_rate) * 100
        
        # Resource score (inverse of max resource usage)
        max_resource_usage = max(
            self.cpu_usage_percent,
            self.memory_usage_percent,
            self.disk_usage_percent
        )
        resource_score = max(0, (100 - max_resource_usage))
        
        # Satisfaction score
        satisfaction_score = 100.0
        if self.average_satisfaction_score is not None:
            satisfaction_score = (self.average_satisfaction_score / 5.0) * 100
        
        total_score = (
            performance_score * performance_weight +
            error_score * error_weight +
            resource_score * resource_weight +
            satisfaction_score * satisfaction_weight
        )
        
        return min(100.0, max(0.0, total_score))
    
    @property
    def is_overloaded(self) -> bool:
        """Check if system is overloaded."""
        return (
            self.cpu_usage_percent > 90 or
            self.memory_usage_percent > 90 or
            self.error_rate > 0.05 or  # 5% error rate
            self.average_response_time > 10.0  # 10 seconds
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe.value,
            "active_connections": self.active_connections,
            "total_connections_today": self.total_connections_today,
            "peak_concurrent_connections": self.peak_concurrent_connections,
            "connection_success_rate": self.connection_success_rate,
            "active_agents": self.active_agents,
            "total_agents": self.total_agents,
            "average_agent_load": self.average_agent_load,
            "overloaded_agents": self.overloaded_agents,
            "average_response_time": self.average_response_time,
            "median_response_time": self.median_response_time,
            "p95_response_time": self.p95_response_time,
            "p99_response_time": self.p99_response_time,
            "requests_per_second": self.requests_per_second,
            "messages_per_second": self.messages_per_second,
            "tokens_per_second": self.tokens_per_second,
            "error_rate": self.error_rate,
            "timeout_rate": self.timeout_rate,
            "connection_drop_rate": self.connection_drop_rate,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_percent": self.memory_usage_percent,
            "disk_usage_percent": self.disk_usage_percent,
            "network_usage_mbps": self.network_usage_mbps,
            "active_voice_calls": self.active_voice_calls,
            "voice_quality_score": self.voice_quality_score,
            "voice_latency_ms": self.voice_latency_ms,
            "total_tokens_today": self.total_tokens_today,
            "estimated_cost_today": self.estimated_cost_today,
            "cost_per_conversation": self.cost_per_conversation,
            "average_satisfaction_score": self.average_satisfaction_score,
            "satisfaction_responses_count": self.satisfaction_responses_count,
            "requests_by_region": self.requests_by_region,
            "api_calls_per_endpoint": self.api_calls_per_endpoint,
            "system_health_score": self.system_health_score,
            "is_overloaded": self.is_overloaded
        }