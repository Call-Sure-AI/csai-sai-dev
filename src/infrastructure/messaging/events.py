# src/infrastructure/messaging/events.py
import asyncio
import logging
from typing import Dict, Any, List, Callable
from datetime import datetime

from core.interfaces.events import IEventPublisher, IEventSubscriber, DomainEvent

logger = logging.getLogger(__name__)

class InMemoryEventBus(IEventPublisher, IEventSubscriber):
    """In-memory event bus implementation"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish domain event"""
        try:
            handlers = self.subscribers.get(event.event_type, [])
            
            # Execute all handlers concurrently
            if handlers:
                tasks = [handler(event) for handler in handlers]
                await asyncio.gather(*tasks, return_exceptions=True)
                
            logger.debug(f"Published event: {event.event_type} to {len(handlers)} handlers")
            
        except Exception as e:
            logger.error(f"Error publishing event {event.event_type}: {e}")
    
    async def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler to {event_type}")
    
    async def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from event type"""
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from {event_type}")
            except ValueError:
                pass  # Handler not in list
    
    def get_subscriber_count(self, event_type: str) -> int:
        """Get number of subscribers for event type"""
        return len(self.subscribers.get(event_type, []))

# Global event bus instance
event_bus = InMemoryEventBus()