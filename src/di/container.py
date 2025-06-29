# src/di/container.py
"""Dependency Injection Container using dependency-injector"""

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide

# Core services
from application.services.connection_service import ConnectionService
from application.services.conversation_service import ConversationService
from application.services.voice_service import VoiceService
from application.services.analytics_service import AnalyticsService

# Infrastructure
from infrastructure.database.repositories.client_repository import ClientRepository
from infrastructure.database.repositories.conversation_repository import ConversationRepository
from infrastructure.database.repositories.analytics_repository import AnalyticsRepository
from infrastructure.database.repositories.agent_repository import AgentRepository

# External services
from infrastructure.external.deepgram_service import DeepgramSTTService
from infrastructure.external.elevenlabs_service import ElevenLabsTTSService
from infrastructure.external.openai_service import OpenAIService
from infrastructure.external.qdrant_service import QdrantVectorService

# Configuration
from config.settings_manager import SettingsManager
from database.config import get_db

class Container(containers.DeclarativeContainer):
    """Application dependency injection container"""
    
    # Configuration
    config = providers.Configuration()
    settings = providers.Singleton(SettingsManager)
    
    # Database
    database_session = providers.Resource(get_db)
    
    # Repositories
    client_repository = providers.Factory(
        ClientRepository,
        session=database_session
    )
    
    conversation_repository = providers.Factory(
        ConversationRepository,
        session=database_session
    )
    
    analytics_repository = providers.Factory(
        AnalyticsRepository,
        session=database_session
    )
    
    agent_repository = providers.Factory(
        AgentRepository,
        session=database_session
    )
    
    # External services
    deepgram_service = providers.Singleton(
        DeepgramSTTService,
        api_key=settings.provided.settings.deepgram_api_key
    )
    
    elevenlabs_service = providers.Singleton(
        ElevenLabsTTSService,
        api_key=settings.provided.settings.eleven_labs_api_key
    )
    
    openai_service = providers.Singleton(
        OpenAIService,
        api_key=settings.provided.settings.openai_api_key
    )
    
    qdrant_service = providers.Singleton(
        QdrantVectorService,
        host=settings.provided.settings.qdrant_host,
        port=settings.provided.settings.qdrant_port,
        api_key=settings.provided.settings.qdrant_api_key
    )
    
    # Application services
    analytics_service = providers.Factory(
        AnalyticsService,
        analytics_repository=analytics_repository
    )
    
    connection_service = providers.Factory(
        ConnectionService,
        client_repository=client_repository,
        agent_repository=agent_repository,
        analytics_service=analytics_service,
        agent_service=providers.Self(),  # Circular dependency - will be resolved
        max_connections=settings.provided.settings.max_connections,
        max_requests_per_minute=settings.provided.settings.max_requests_per_minute
    )
    
    conversation_service = providers.Factory(
        ConversationService,
        conversation_repository=conversation_repository,
        ai_service=openai_service,
        analytics_service=analytics_service,
        agent_service=providers.Self()  # Will be resolved later
    )
    
    voice_service = providers.Factory(
        VoiceService,
        stt_service=deepgram_service,
        tts_service=elevenlabs_service,
        analytics_service=analytics_service
    )


# Global container instance
container = Container()

def configure_container():
    """Configure the container with settings"""
    settings_manager = SettingsManager()
    container.config.from_dict(settings_manager.get_config_dict())