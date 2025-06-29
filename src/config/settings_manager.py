# src/config/settings_manager.py
import os
import logging
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator

logger = logging.getLogger(__name__)

class ApplicationSettings(BaseSettings):
    """Centralized application settings with validation"""
    
    # Application Core
    app_name: str = Field(default="AI Backend", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    reload: bool = Field(default=False, env="RELOAD")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    
    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # External APIs
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    deepgram_api_key: str = Field(..., env="DEEPGRAM_API_KEY")
    eleven_labs_api_key: str = Field(..., env="ELEVEN_LABS_API_KEY")
    
    # Qdrant Vector Store
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_https: bool = Field(default=False, env="QDRANT_HTTPS")
    
    # Application Limits
    max_connections: int = Field(default=1000, env="MAX_CONNECTIONS")
    max_requests_per_minute: int = Field(default=60, env="MAX_REQUESTS_PER_MINUTE")
    max_message_length: int = Field(default=5000, env="MAX_MESSAGE_LENGTH")
    max_voice_duration: int = Field(default=3600, env="MAX_VOICE_DURATION")  # seconds
    
    # Processing Configuration
    processing_timeout: float = Field(default=30.0, env="PROCESSING_TIMEOUT")
    max_concurrent_processes: int = Field(default=20, env="MAX_CONCURRENT_PROCESSES")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=8001, env="METRICS_PORT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Feature Flags
    enable_voice_calls: bool = Field(default=True, env="ENABLE_VOICE_CALLS")
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    
    @validator('cors_origins')
    def validate_cors_origins(cls, v):
        """Convert CORS origins string to list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment"""
        valid_envs = ['development', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f'Environment must be one of: {valid_envs}')
        return v.lower()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class SettingsManager:
    """Singleton settings manager"""
    
    _instance: Optional['SettingsManager'] = None
    _settings: Optional[ApplicationSettings] = None
    
    def __new__(cls) -> 'SettingsManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def settings(self) -> ApplicationSettings:
        """Get application settings"""
        if self._settings is None:
            try:
                self._settings = ApplicationSettings()
                self._configure_logging()
                logger.info(f"Settings loaded for environment: {self._settings.environment}")
            except Exception as e:
                logger.error(f"Failed to load settings: {e}")
                raise
        return self._settings
    
    def _configure_logging(self) -> None:
        """Configure logging based on settings"""
        if not self._settings:
            return
        
        logging.basicConfig(
            level=getattr(logging, self._settings.log_level),
            format=self._settings.log_format,
            filename=self._settings.log_file
        )
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        config = self.settings.dict()
        
        # Mask sensitive values
        sensitive_keys = [
            'secret_key', 'database_url', 'openai_api_key', 
            'deepgram_api_key', 'eleven_labs_api_key', 'redis_password'
        ]
        
        for key in sensitive_keys:
            if key in config and config[key]:
                config[key] = "*" * 8
        
        return config
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        try:
            settings = self.settings
            
            # Check required external services
            if not settings.openai_api_key:
                issues.append("OpenAI API key is required")
            
            if not settings.deepgram_api_key:
                issues.append("Deepgram API key is required")
            
            if not settings.database_url:
                issues.append("Database URL is required")
            
            # Check logical constraints
            if settings.max_connections <= 0:
                issues.append("Max connections must be positive")
            
            if settings.max_requests_per_minute <= 0:
                issues.append("Max requests per minute must be positive")
            
            if settings.processing_timeout <= 0:
                issues.append("Processing timeout must be positive")
            
            # Environment-specific checks
            if settings.environment == "production":
                if settings.debug:
                    issues.append("Debug mode should be disabled in production")
                
                if settings.secret_key == "change-this-in-production":
                    issues.append("Secret key must be changed in production")
            
        except Exception as e:
            issues.append(f"Configuration validation error: {str(e)}")
        
        return issues
    
    def reload_settings(self) -> None:
        """Reload settings from environment"""
        self._settings = None
        _ = self.settings  # Trigger reload

# Global settings manager instance
settings_manager = SettingsManager()

def get_settings() -> ApplicationSettings:
    """Get global settings instance"""
    return settings_manager.settings

def validate_startup_configuration() -> None:
    """Validate configuration on startup"""
    issues = settings_manager.validate_configuration()
    
    if issues:
        error_message = "Configuration validation failed:\n" + "\n".join(f"- {issue}" for issue in issues)
        logger.error(error_message)
        raise ValueError(error_message)
    
    logger.info("Configuration validation passed")