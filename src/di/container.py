# src/di/container.py
"""
Main dependency injection container for the application.
"""

import logging
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide

# Import all services and repositories
from application.services.connection_service import ConnectionService
from application.services.conversation_service import ConversationService
from application.services.voice_service import VoiceService
from application.services.analytics_service import AnalyticsService

from infrastructure.database.repositories.client_repository import ClientRepository
from infrastructure.database.repositories.conversation_repository import ConversationRepository
from infrastructure.database.repositories.agent_repository import AgentRepository
from infrastructure.database.repositories.analytics_repository import AnalyticsRepository

from infrastructure.external.openai_service import OpenAIService
from infrastructure.external.deepgram_service import DeepgramService
from infrastructure.external.elevenlabs_service import ElevenLabsService

from infrastructure.storage.s3_storage import S3StorageService
from infrastructure.storage.redis_cache import RedisCacheService

from database.config import get_db
from config.settings import settings

logger = logging.getLogger(__name__)

class Container(containers.DeclarativeContainer):
    """Main dependency injection container."""
    
    # Configuration
    config = providers.Configuration()
    
    # Database
    database_session = providers.Singleton(
        get_db
    )
    
    # External Services Configuration
    openai_config = providers.Object({
        "api_key": settings.OPENAI_API_KEY,
        "default_model": settings.OPENAI_MODEL
    })
    
    deepgram_config = providers.Object({
        "api_key": settings.DEEPGRAM_API_KEY,
        "default_model": "nova-2"
    })
    
    elevenlabs_config = providers.Object({
        "api_key": settings.ELEVEN_LABS_API_KEY,
        "default_voice": settings.VOICE_ID
    })
    
    s3_config = providers.Object({
        "bucket_name": getattr(settings, 'S3_BUCKET_NAME', 'ai-voice-storage'),
        "aws_access_key_id": getattr(settings, 'AWS_ACCESS_KEY_ID', None),
        "aws_secret_access_key": getattr(settings, 'AWS_SECRET_ACCESS_KEY', None),
        "region_name": getattr(settings, 'S3_REGION', 'us-east-1')
    })
    
    redis_config = providers.Object({
        "host": settings.REDIS_HOST,
        "port": settings.REDIS_PORT,
        "password": settings.REDIS_PASSWORD,
        "db": 0
    })
    
    # External Services
    openai_service = providers.Singleton(
        OpenAIService,
        api_key=openai_config.provided.api_key,
        default_model=openai_config.provided.default_model
    )
    
    deepgram_service = providers.Singleton(
        DeepgramService,
        api_key=deepgram_config.provided.api_key,
        default_model=deepgram_config.provided.default_model
    )
    
    elevenlabs_service = providers.Singleton(
        ElevenLabsService,
        api_key=elevenlabs_config.provided.api_key,
        default_voice=elevenlabs_config.provided.default_voice
    )
    
    s3_service = providers.Singleton(
        S3StorageService,
        bucket_name=s3_config.provided.bucket_name,
        aws_access_key_id=s3_config.provided.aws_access_key_id,
        aws_secret_access_key=s3_config.provided.aws_secret_access_key,
        region_name=s3_config.provided.region_name
    )
    
    redis_service = providers.Singleton(
        RedisCacheService,
        host=redis_config.provided.host,
        port=redis_config.provided.port,
        password=redis_config.provided.password,
        db=redis_config.provided.db
    )
    
    # Repositories
    client_repository = providers.Factory(
        ClientRepository,
        session=database_session.provided.return_value
    )
    
    conversation_repository = providers.Factory(
        ConversationRepository,
        session=database_session.provided.return_value
    )
    
    agent_repository = providers.Factory(
        AgentRepository,
        session=database_session.provided.return_value
    )
    
    analytics_repository = providers.Factory(
        AnalyticsRepository,
        session=database_session.provided.return_value
    )
    
    # Application Services
    connection_service = providers.Factory(
        ConnectionService,
        client_repository=client_repository,
        max_connections=getattr(settings, 'MAX_CONNECTIONS', 1000),
        connection_timeout=300,
        heartbeat_interval=30
    )
    
    conversation_service = providers.Factory(
        ConversationService,
        conversation_repository=conversation_repository,
        language_model_service=openai_service,
        default_system_prompt="You are a helpful AI assistant for voice calls.",
        max_context_length=4000
    )
    
    voice_service = providers.Factory(
        VoiceService,
        stt_service=deepgram_service,
        tts_service=elevenlabs_service,
        max_call_duration=3600
    )
    
    analytics_service = providers.Factory(
        AnalyticsService,
        analytics_repository=analytics_repository
    )
    
    @classmethod
    def wire_modules(cls, modules: list = None):
        """Wire dependency injection to modules."""
        if modules is None:
            modules = [
                "routes.webrtc_handlers",
                "routes.voice_routes_handlers", 
                "routes.admin_routes_handlers",
                "routes.healthcheck_handlers"
            ]
        
        container = cls()
        container.wire(modules=modules)
        logger.info(f"Wired DI container to modules: {modules}")
        return container