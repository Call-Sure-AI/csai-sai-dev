# src/di/providers.py
"""Service providers for dependency injection"""

from typing import Type, Any, Dict, Optional
from abc import ABC, abstractmethod

class ServiceProvider(ABC):
    """Base service provider interface"""
    
    @abstractmethod
    def register_services(self, container: Any) -> None:
        """Register services with the container"""
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the provider with settings"""
        pass

class DatabaseProvider(ServiceProvider):
    """Provider for database-related services"""
    
    def register_services(self, container: Any) -> None:
        """Register database services"""
        # Register repositories
        container.wire(
            modules=[
                "infrastructure.database.repositories.client_repository",
                "infrastructure.database.repositories.conversation_repository",
                "infrastructure.database.repositories.analytics_repository",
                "infrastructure.database.repositories.agent_repository"
            ]
        )
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure database settings"""
        # Set database configuration
        pass

class ExternalServiceProvider(ServiceProvider):
    """Provider for external service integrations"""
    
    def register_services(self, container: Any) -> None:
        """Register external services"""
        container.wire(
            modules=[
                "infrastructure.external.deepgram_service",
                "infrastructure.external.elevenlabs_service",
                "infrastructure.external.openai_service",
                "infrastructure.external.qdrant_service"
            ]
        )
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure external service settings"""
        # Set API keys and endpoints
        pass

class ApplicationServiceProvider(ServiceProvider):
    """Provider for application services"""
    
    def register_services(self, container: Any) -> None:
        """Register application services"""
        container.wire(
            modules=[
                "application.services.connection_service",
                "application.services.conversation_service",
                "application.services.voice_service",
                "application.services.analytics_service"
            ]
        )
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure application settings"""
        # Set application-specific configuration
        pass