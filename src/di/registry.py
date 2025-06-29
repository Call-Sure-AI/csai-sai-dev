# src/di/registry.py
"""Service registry for managing service instances"""

from typing import Dict, Any, Type, Optional, TypeVar, Generic
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class IServiceRegistry(ABC):
    """Interface for service registry"""
    
    @abstractmethod
    def register(self, service_type: Type[T], instance: T) -> None:
        """Register service instance"""
        pass
    
    @abstractmethod
    def get(self, service_type: Type[T]) -> Optional[T]:
        """Get service instance"""
        pass
    
    @abstractmethod
    def is_registered(self, service_type: Type[T]) -> bool:
        """Check if service is registered"""
        pass

class ServiceRegistry(IServiceRegistry):
    """Simple service registry implementation"""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
    
    def register(self, service_type: Type[T], instance: T) -> None:
        """Register service instance"""
        self._services[service_type] = instance
        logger.debug(f"Registered service: {service_type.__name__}")
    
    def register_singleton(self, service_type: Type[T], instance: T) -> None:
        """Register singleton service instance"""
        self._singletons[service_type] = instance
        logger.debug(f"Registered singleton: {service_type.__name__}")
    
    def get(self, service_type: Type[T]) -> Optional[T]:
        """Get service instance"""
        # Check singletons first
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        # Check regular services
        return self._services.get(service_type)
    
    def is_registered(self, service_type: Type[T]) -> bool:
        """Check if service is registered"""
        return service_type in self._services or service_type in self._singletons
    
    def unregister(self, service_type: Type[T]) -> None:
        """Unregister service"""
        self._services.pop(service_type, None)
        self._singletons.pop(service_type, None)
        logger.debug(f"Unregistered service: {service_type.__name__}")
    
    def clear(self) -> None:
        """Clear all registered services"""
        self._services.clear()
        self._singletons.clear()
        logger.info("Cleared all registered services")
    
    def get_all_registered(self) -> Dict[str, Type]:
        """Get all registered service types"""
        all_services = {}
        
        for service_type in self._services.keys():
            all_services[service_type.__name__] = service_type
        
        for service_type in self._singletons.keys():
            all_services[f"{service_type.__name__} (singleton)"] = service_type
        
        return all_services

# Global registry instance
service_registry = ServiceRegistry()

def register_core_services():
    """Register core application services with the registry"""
    from core.interfaces.services import (
        IConnectionService, IConversationService, 
        IVoiceService, IAnalyticsService
    )
    from application.services.connection_service import ConnectionService
    from application.services.conversation_service import ConversationService
    from application.services.voice_service import VoiceService
    from application.services.analytics_service import AnalyticsService
    
    # This would typically be done through the DI container
    # but shown here for demonstration
    logger.info("Registering core services...")
    
    # Note: In practice, these would be instantiated with proper dependencies
    # service_registry.register(IConnectionService, ConnectionService(...))
    # service_registry.register(IConversationService, ConversationService(...))
    # service_registry.register(IVoiceService, VoiceService(...))
    # service_registry.register(IAnalyticsService, AnalyticsService(...))

class ServiceLocator:
    """Service locator pattern implementation"""
    
    def __init__(self, registry: IServiceRegistry):
        self.registry = registry
    
    def get_service(self, service_type: Type[T]) -> T:
        """Get service with error handling"""
        service = self.registry.get(service_type)
        if service is None:
            raise ValueError(f"Service not registered: {service_type.__name__}")
        return service
    
    def try_get_service(self, service_type: Type[T]) -> Optional[T]:
        """Try to get service without raising exception"""
        return self.registry.get(service_type)

# Global service locator
service_locator = ServiceLocator(service_registry)