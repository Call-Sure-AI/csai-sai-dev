"""
Event interfaces for the messaging and event-driven architecture.

These interfaces define contracts for event publishing, subscribing,
and handling in the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Awaitable
from datetime import datetime


class IEvent(ABC):
    """Base interface for all events."""
    
    @property
    @abstractmethod
    def event_id(self) -> str:
        """Unique identifier for the event."""
        pass
    
    @property
    @abstractmethod
    def event_type(self) -> str:
        """Type/name of the event."""
        pass
    
    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        """When the event occurred."""
        pass
    
    @property
    @abstractmethod
    def source(self) -> str:
        """Source that generated the event."""
        pass
    
    @property
    @abstractmethod
    def data(self) -> Dict[str, Any]:
        """Event payload data."""
        pass


class IEventPublisher(ABC):
    """Interface for publishing events."""
    
    @abstractmethod
    async def publish(
        self,
        event: IEvent,
        routing_key: Optional[str] = None
    ) -> bool:
        """Publish an event."""
        pass
    
    @abstractmethod
    async def publish_batch(
        self,
        events: List[IEvent],
        routing_key: Optional[str] = None
    ) -> int:
        """Publish multiple events and return number successfully published."""
        pass
    
    @abstractmethod
    async def publish_delayed(
        self,
        event: IEvent,
        delay_seconds: int,
        routing_key: Optional[str] = None
    ) -> str:
        """Publish an event with a delay and return scheduled event ID."""
        pass
    
    @abstractmethod
    async def cancel_delayed_event(self, scheduled_event_id: str) -> bool:
        """Cancel a delayed event."""
        pass


class IEventSubscriber(ABC):
    """Interface for subscribing to events."""
    
    @abstractmethod
    async def subscribe(
        self,
        event_type: str,
        handler: 'IEventHandler',
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Subscribe to events of a specific type and return subscription ID."""
        pass
    
    @abstractmethod
    async def subscribe_pattern(
        self,
        pattern: str,
        handler: 'IEventHandler',
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Subscribe to events matching a pattern."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        pass
    
    @abstractmethod
    async def unsubscribe_all(self, handler: 'IEventHandler') -> int:
        """Unsubscribe all subscriptions for a handler."""
        pass
    
    @abstractmethod
    async def get_subscriptions(self) -> List[Dict[str, Any]]:
        """Get all active subscriptions."""
        pass


class IEventHandler(ABC):
    """Interface for handling events."""
    
    @abstractmethod
    async def handle(self, event: IEvent) -> bool:
        """Handle an event and return success status."""
        pass
    
    @abstractmethod
    async def can_handle(self, event: IEvent) -> bool:
        """Check if this handler can process the event."""
        pass
    
    @property
    @abstractmethod
    def handler_name(self) -> str:
        """Unique name for this handler."""
        pass


class IEventStore(ABC):
    """Interface for event storage and retrieval."""
    
    @abstractmethod
    async def store_event(self, event: IEvent) -> None:
        """Store an event."""
        pass
    
    @abstractmethod
    async def get_events(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[IEvent]:
        """Retrieve events with filters."""
        pass
    
    @abstractmethod
    async def get_event_stream(
        self,
        entity_id: str,
        entity_type: str
    ) -> List[IEvent]:
        """Get event stream for a specific entity."""
        pass
    
    @abstractmethod
    async def replay_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[str]] = None
    ) -> List[IEvent]:
        """Replay events from a time period."""
        pass
    
    @abstractmethod
    async def cleanup_old_events(self, retention_days: int = 30) -> int:
        """Clean up old events and return number deleted."""
        pass


class IEventBus(ABC):
    """Interface for event bus coordination."""
    
    @abstractmethod
    async def start(self) -> None:
        """Start the event bus."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the event bus."""
        pass
    
    @abstractmethod
    async def publish(
        self,
        event: IEvent,
        routing_key: Optional[str] = None
    ) -> bool:
        """Publish an event through the bus."""
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        event_type: str,
        handler: IEventHandler,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Subscribe to events through the bus."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        pass


# Domain Event Types
class IConnectionEvent(IEvent):
    """Interface for connection-related events."""
    
    @property
    @abstractmethod
    def client_id(self) -> str:
        """Client ID associated with the event."""
        pass
    
    @property
    @abstractmethod
    def session_id(self) -> str:
        """Session ID associated with the event."""
        pass


class IConversationEvent(IEvent):
    """Interface for conversation-related events."""
    
    @property
    @abstractmethod
    def conversation_id(self) -> str:
        """Conversation ID associated with the event."""
        pass
    
    @property
    @abstractmethod
    def client_id(self) -> str:
        """Client ID associated with the event."""
        pass
    
    @property
    @abstractmethod
    def agent_id(self) -> Optional[str]:
        """Agent ID associated with the event."""
        pass


class IVoiceEvent(IEvent):
    """Interface for voice-related events."""
    
    @property
    @abstractmethod
    def client_id(self) -> str:
        """Client ID associated with the event."""
        pass
    
    @property
    @abstractmethod
    def call_id(self) -> str:
        """Voice call ID associated with the event."""
        pass


class IAgentEvent(IEvent):
    """Interface for agent-related events."""
    
    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Agent ID associated with the event."""
        pass


class ISystemEvent(IEvent):
    """Interface for system-level events."""
    
    @property
    @abstractmethod
    def component(self) -> str:
        """System component that generated the event."""
        pass
    
    @property
    @abstractmethod
    def severity(self) -> str:
        """Event severity level."""
        pass


# Event Handler Function Type
EventHandlerFunc = Callable[[IEvent], Awaitable[bool]]


class IEventProcessor(ABC):
    """Interface for event processing and routing."""
    
    @abstractmethod
    async def process_event(self, event: IEvent) -> bool:
        """Process a single event."""
        pass
    
    @abstractmethod
    async def process_batch(self, events: List[IEvent]) -> List[bool]:
        """Process multiple events and return success status for each."""
        pass
    
    @abstractmethod
    async def add_middleware(
        self,
        middleware: Callable[[IEvent], Awaitable[IEvent]]
    ) -> None:
        """Add middleware to the event processing pipeline."""
        pass
    
    @abstractmethod
    async def remove_middleware(
        self,
        middleware: Callable[[IEvent], Awaitable[IEvent]]
    ) -> bool:
        """Remove middleware from the processing pipeline."""
        pass
    
    @abstractmethod
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get event processing statistics."""
        pass


class IEventFilter(ABC):
    """Interface for filtering events."""
    
    @abstractmethod
    def matches(self, event: IEvent) -> bool:
        """Check if event matches the filter criteria."""
        pass
    
    @abstractmethod
    def get_filter_criteria(self) -> Dict[str, Any]:
        """Get the filter criteria."""
        pass


class IEventTransformer(ABC):
    """Interface for transforming events."""
    
    @abstractmethod
    async def transform(self, event: IEvent) -> IEvent:
        """Transform an event and return the modified event."""
        pass
    
    @abstractmethod
    def can_transform(self, event: IEvent) -> bool:
        """Check if this transformer can handle the event."""
        pass


class IEventAggregator(ABC):
    """Interface for aggregating events."""
    
    @abstractmethod
    async def aggregate(
        self,
        events: List[IEvent],
        window_size: int = 60
    ) -> Optional[IEvent]:
        """Aggregate multiple events into a single event."""
        pass
    
    @abstractmethod
    async def add_event(self, event: IEvent) -> Optional[IEvent]:
        """Add event to aggregation buffer and return aggregated event if ready."""
        pass
    
    @abstractmethod
    async def flush(self) -> List[IEvent]:
        """Flush any pending aggregated events."""
        pass


class IEventScheduler(ABC):
    """Interface for scheduling event processing."""
    
    @abstractmethod
    async def schedule_event(
        self,
        event: IEvent,
        schedule_time: datetime
    ) -> str:
        """Schedule an event for future processing."""
        pass
    
    @abstractmethod
    async def schedule_recurring(
        self,
        event_template: IEvent,
        interval_seconds: int,
        max_occurrences: Optional[int] = None
    ) -> str:
        """Schedule a recurring event."""
        pass
    
    @abstractmethod
    async def cancel_scheduled(self, schedule_id: str) -> bool:
        """Cancel a scheduled event."""
        pass
    
    @abstractmethod
    async def get_scheduled_events(self) -> List[Dict[str, Any]]:
        """Get all scheduled events."""
        pass


class IEventMetrics(ABC):
    """Interface for event system metrics."""
    
    @abstractmethod
    async def record_event_published(
        self,
        event_type: str,
        success: bool,
        latency_ms: float
    ) -> None:
        """Record event publication metrics."""
        pass
    
    @abstractmethod
    async def record_event_processed(
        self,
        event_type: str,
        handler_name: str,
        success: bool,
        processing_time_ms: float
    ) -> None:
        """Record event processing metrics."""
        pass
    
    @abstractmethod
    async def get_event_throughput(
        self,
        time_window_minutes: int = 5
    ) -> Dict[str, float]:
        """Get event throughput metrics."""
        pass
    
    @abstractmethod
    async def get_handler_performance(
        self,
        handler_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get event handler performance metrics."""
        pass
    
    @abstractmethod
    async def get_error_rates(self) -> Dict[str, float]:
        """Get event processing error rates."""
        pass


class IEventSaga(ABC):
    """Interface for event-driven sagas/workflows."""
    
    @abstractmethod
    async def start_saga(
        self,
        saga_id: str,
        initial_event: IEvent,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Start a new saga."""
        pass
    
    @abstractmethod
    async def handle_event(
        self,
        saga_id: str,
        event: IEvent
    ) -> bool:
        """Handle an event in the context of a saga."""
        pass
    
    @abstractmethod
    async def complete_saga(self, saga_id: str) -> bool:
        """Mark a saga as completed."""
        pass
    
    @abstractmethod
    async def compensate_saga(self, saga_id: str) -> bool:
        """Run compensation actions for a failed saga."""
        pass
    
    @abstractmethod
    async def get_saga_status(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a saga."""
        pass


class IEventProjector(ABC):
    """Interface for event projections/read models."""
    
    @abstractmethod
    async def project(self, event: IEvent) -> bool:
        """Project an event to update read models."""
        pass
    
    @abstractmethod
    async def rebuild_projection(
        self,
        projection_name: str,
        start_time: Optional[datetime] = None
    ) -> bool:
        """Rebuild a projection from event history."""
        pass
    
    @abstractmethod
    async def get_projection_status(self, projection_name: str) -> Dict[str, Any]:
        """Get status of a projection."""
        pass


class IEventReplay(ABC):
    """Interface for event replay functionality."""
    
    @abstractmethod
    async def replay_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[str]] = None,
        target_handlers: Optional[List[str]] = None
    ) -> int:
        """Replay events and return number of events replayed."""
        pass
    
    @abstractmethod
    async def replay_entity_events(
        self,
        entity_id: str,
        entity_type: str,
        target_handlers: Optional[List[str]] = None
    ) -> int:
        """Replay all events for a specific entity."""
        pass
    
    @abstractmethod
    async def create_replay_job(
        self,
        replay_config: Dict[str, Any]
    ) -> str:
        """Create an asynchronous replay job."""
        pass
    
    @abstractmethod
    async def get_replay_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a replay job."""
        pass


class IEventSnapshot(ABC):
    """Interface for event sourcing snapshots."""
    
    @abstractmethod
    async def create_snapshot(
        self,
        entity_id: str,
        entity_type: str,
        entity_state: Dict[str, Any],
        event_version: int
    ) -> bool:
        """Create a snapshot of entity state."""
        pass
    
    @abstractmethod
    async def get_snapshot(
        self,
        entity_id: str,
        entity_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get the latest snapshot for an entity."""
        pass
    
    @abstractmethod
    async def cleanup_old_snapshots(
        self,
        entity_id: str,
        entity_type: str,
        keep_count: int = 5
    ) -> int:
        """Clean up old snapshots and return number deleted."""
        pass