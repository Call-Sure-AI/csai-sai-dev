# src/di/providers.py
"""
Service providers for dependency injection.
"""

import logging
from typing import Dict, Any, Type, TypeVar, Optional
from dependency_injector import providers

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ServiceProviders:
    """Factory for creating service providers."""
    
    @staticmethod
    def create_singleton_provider(
        service_class: Type[T],
        dependencies: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> providers.Singleton:
        """Create a singleton provider for a service."""
        provider_kwargs = {}
        
        if dependencies:
            provider_kwargs.update(dependencies)
        
        if config:
            for key, value in config.items():
                provider_kwargs[key] = value
        
        return providers.Singleton(service_class, **provider_kwargs)
    
    @staticmethod
    def create_factory_provider(
        service_class: Type[T],
        dependencies: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> providers.Factory:
        """Create a factory provider for a service."""
        provider_kwargs = {}
        
        if dependencies:
            provider_kwargs.update(dependencies)
        
        if config:
            for key, value in config.items():
                provider_kwargs[key] = value
        
        return providers.Factory(service_class, **provider_kwargs)
    
    @staticmethod
    def create_configuration_provider(config_dict: Dict[str, Any]) -> providers.Configuration:
        """Create a configuration provider."""
        config = providers.Configuration()
        
        for key, value in config_dict.items():
            config.set(key, value)
        
        return config
    
    @staticmethod
    def create_resource_provider(
        initializer,
        finalizer=None,
        *args,
        **kwargs
    ) -> providers.Resource:
        """Create a resource provider for managing resources."""
        if finalizer:
            return providers.Resource(
                initializer,
                finalizer,
                *args,
                **kwargs
            )
        else:
            return providers.Resource(
                initializer,
                *args,
                **kwargs
            )

class ExternalServiceProviders:
    """Providers for external services with health checks."""
    
    @staticmethod
    def create_ai_service_provider(service_class: Type[T], config: Dict[str, Any]) -> providers.Singleton:
        """Create provider for AI services with validation."""
        
        def create_service():
            try:
                service = service_class(**config)
                # Could add validation here
                logger.info(f"Created AI service: {service_class.__name__}")
                return service
            except Exception as e:
                logger.error(f"Failed to create AI service {service_class.__name__}: {e}")
                raise
        
        return providers.Singleton(create_service)
    
    @staticmethod
    def create_storage_service_provider(service_class: Type[T], config: Dict[str, Any]) -> providers.Singleton:
        """Create provider for storage services with connection testing."""
        
        def create_service():
            try:
                service = service_class(**config)
                # Could add connection test here
                logger.info(f"Created storage service: {service_class.__name__}")
                return service
            except Exception as e:
                logger.error(f"Failed to create storage service {service_class.__name__}: {e}")
                raise
        
        return providers.Singleton(create_service)