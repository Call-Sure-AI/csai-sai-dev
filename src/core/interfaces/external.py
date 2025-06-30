"""
External service interfaces for third-party integrations.

These interfaces define contracts for external services like AI models,
storage, communication services, and other third-party APIs.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, AsyncGenerator, Union
from datetime import datetime
import asyncio


class ILanguageModelService(ABC):
    """Interface for language model services (OpenAI, Anthropic, etc.)."""
    
    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text completion."""
        pass
    
    @abstractmethod
    async def generate_text_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text completion with streaming."""
        pass
    
    @abstractmethod
    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion."""
        pass
    
    @abstractmethod
    async def generate_chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate chat completion with streaming."""
        pass
    
    @abstractmethod
    async def embed_text(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """Generate text embeddings."""
        pass
    
    @abstractmethod
    async def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        pass
    
    @abstractmethod
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        pass
    
    @abstractmethod
    async def validate_api_key(self) -> bool:
        """Validate the API key."""
        pass


class ITextToSpeechService(ABC):
    """Interface for text-to-speech services."""
    
    @abstractmethod
    async def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        output_format: str = "mp3",
        **kwargs
    ) -> bytes:
        """Convert text to speech audio."""
        pass
    
    @abstractmethod
    async def synthesize_speech_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        output_format: str = "mp3",
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """Convert text to speech with streaming."""
        pass
    
    @abstractmethod
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices."""
        pass
    
    @abstractmethod
    async def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """Get information about a specific voice."""
        pass
    
    @abstractmethod
    async def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        pass
    
    @abstractmethod
    async def clone_voice(
        self,
        name: str,
        audio_samples: List[bytes],
        description: Optional[str] = None
    ) -> str:
        """Clone a voice from audio samples."""
        pass


class ISpeechToTextService(ABC):
    """Interface for speech-to-text services."""
    
    @abstractmethod
    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        language: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe audio to text."""
        pass
    
    @abstractmethod
    async def transcribe_audio_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        audio_format: str = "wav",
        language: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Transcribe audio stream in real-time."""
        pass
    
    @abstractmethod
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        pass
    
    @abstractmethod
    async def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available transcription models."""
        pass


class IStorageService(ABC):
    """Interface for cloud storage services."""
    
    @abstractmethod
    async def upload_file(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload file and return URL."""
        pass
    
    @abstractmethod
    async def download_file(self, key: str) -> bytes:
        """Download file by key."""
        pass
    
    @abstractmethod
    async def delete_file(self, key: str) -> bool:
        """Delete file by key."""
        pass
    
    @abstractmethod
    async def file_exists(self, key: str) -> bool:
        """Check if file exists."""
        pass
    
    @abstractmethod
    async def get_file_info(self, key: str) -> Dict[str, Any]:
        """Get file metadata."""
        pass
    
    @abstractmethod
    async def list_files(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List files with optional prefix filter."""
        pass
    
    @abstractmethod
    async def generate_presigned_url(
        self,
        key: str,
        expiration_seconds: int = 3600,
        method: str = "GET"
    ) -> str:
        """Generate presigned URL for direct access."""
        pass
    
    @abstractmethod
    async def copy_file(self, source_key: str, dest_key: str) -> bool:
        """Copy file from source to destination."""
        pass


class INotificationService(ABC):
    """Interface for notification services."""
    
    @abstractmethod
    async def send_email(
        self,
        to_addresses: List[str],
        subject: str,
        body: str,
        is_html: bool = False,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Send email notification."""
        pass
    
    @abstractmethod
    async def send_sms(
        self,
        phone_number: str,
        message: str
    ) -> bool:
        """Send SMS notification."""
        pass
    
    @abstractmethod
    async def send_push_notification(
        self,
        device_token: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send push notification."""
        pass
    
    @abstractmethod
    async def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ) -> bool:
        """Send webhook notification."""
        pass
    
    @abstractmethod
    async def send_slack_message(
        self,
        channel: str,
        message: str,
        webhook_url: Optional[str] = None
    ) -> bool:
        """Send Slack message."""
        pass


class ITelephonyService(ABC):
    """Interface for telephony services (Twilio, Exotel, etc.)."""
    
    @abstractmethod
    async def make_call(
        self,
        from_number: str,
        to_number: str,
        webhook_url: str,
        record_call: bool = False
    ) -> str:
        """Initiate a phone call and return call ID."""
        pass
    
    @abstractmethod
    async def answer_call(
        self,
        call_id: str,
        actions: List[Dict[str, Any]]
    ) -> bool:
        """Answer an incoming call with TwiML/actions."""
        pass
    
    @abstractmethod
    async def hangup_call(self, call_id: str) -> bool:
        """Hang up a call."""
        pass
    
    @abstractmethod
    async def get_call_status(self, call_id: str) -> Dict[str, Any]:
        """Get current call status."""
        pass
    
    @abstractmethod
    async def get_call_recording(self, call_id: str) -> Optional[str]:
        """Get call recording URL."""
        pass
    
    @abstractmethod
    async def send_sms(
        self,
        from_number: str,
        to_number: str,
        message: str
    ) -> str:
        """Send SMS and return message ID."""
        pass
    
    @abstractmethod
    async def get_available_numbers(
        self,
        country_code: str = "US"
    ) -> List[Dict[str, Any]]:
        """Get available phone numbers for purchase."""
        pass


class ICacheService(ABC):
    """Interface for caching services (Redis, Memcached, etc.)."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values by keys."""
        pass
    
    @abstractmethod
    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple key-value pairs."""
        pass
    
    @abstractmethod
    async def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys and return count deleted."""
        pass
    
    @abstractmethod
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment numeric value."""
        pass
    
    @abstractmethod
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for key."""
        pass
    
    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        pass


class IVectorDatabaseService(ABC):
    """Interface for vector database services."""
    
    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimension: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new vector collection."""
        pass
    
    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """Delete a vector collection."""
        pass
    
    @abstractmethod
    async def insert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Insert vectors into collection."""
        pass
    
    @abstractmethod
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def update_vector(
        self,
        collection_name: str,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a specific vector."""
        pass
    
    @abstractmethod
    async def delete_vector(self, collection_name: str, vector_id: str) -> bool:
        """Delete a specific vector."""
        pass
    
    @abstractmethod
    async def get_collection_info(self, name: str) -> Dict[str, Any]:
        """Get collection information."""
        pass


class IMonitoringService(ABC):
    """Interface for monitoring and observability services."""
    
    @abstractmethod
    async def send_metric(
        self,
        name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Send a metric data point."""
        pass
    
    @abstractmethod
    async def send_event(
        self,
        title: str,
        text: str,
        alert_type: str = "info",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Send an event."""
        pass
    
    @abstractmethod
    async def start_trace(self, operation_name: str) -> str:
        """Start a distributed trace and return trace ID."""
        pass
    
    @abstractmethod
    async def end_trace(
        self,
        trace_id: str,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """End a distributed trace."""
        pass
    
    @abstractmethod
    async def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an error with context."""
        pass
    
    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Check monitoring service health."""
        pass


class IPaymentService(ABC):
    """Interface for payment processing services."""
    
    @abstractmethod
    async def create_customer(
        self,
        customer_data: Dict[str, Any]
    ) -> str:
        """Create a new customer and return customer ID."""
        pass
    
    @abstractmethod
    async def charge_customer(
        self,
        customer_id: str,
        amount: int,  # Amount in cents
        currency: str = "usd",
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Charge a customer and return payment result."""
        pass
    
    @abstractmethod
    async def create_subscription(
        self,
        customer_id: str,
        plan_id: str,
        trial_days: Optional[int] = None
    ) -> str:
        """Create a subscription and return subscription ID."""
        pass
    
    @abstractmethod
    async def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a subscription."""
        pass
    
    @abstractmethod
    async def get_payment_history(
        self,
        customer_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get payment history for a customer."""
        pass
    
    @abstractmethod
    async def refund_payment(
        self,
        payment_id: str,
        amount: Optional[int] = None
    ) -> Dict[str, Any]:
        """Refund a payment (partial or full)."""
        pass


class ICalendarService(ABC):
    """Interface for calendar integration services."""
    
    @abstractmethod
    async def get_events(
        self,
        calendar_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get calendar events in time range."""
        pass
    
    @abstractmethod
    async def create_event(
        self,
        calendar_id: str,
        event_data: Dict[str, Any]
    ) -> str:
        """Create a calendar event and return event ID."""
        pass
    
    @abstractmethod
    async def update_event(
        self,
        calendar_id: str,
        event_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a calendar event."""
        pass
    
    @abstractmethod
    async def delete_event(
        self,
        calendar_id: str,
        event_id: str
    ) -> bool:
        """Delete a calendar event."""
        pass
    
    @abstractmethod
    async def get_available_slots(
        self,
        calendar_id: str,
        start_time: datetime,
        end_time: datetime,
        duration_minutes: int
    ) -> List[Dict[str, Any]]:
        """Get available time slots."""
        pass


class IAuthenticationProvider(ABC):
    """Interface for external authentication providers."""
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate authentication token and return user info."""
        pass
    
    @abstractmethod
    async def refresh_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """Refresh access token."""
        pass
    
    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """Revoke an access token."""
        pass
    
    @abstractmethod
    async def get_user_info(self, token: str) -> Optional[Dict[str, Any]]:
        """Get user information from token."""
        pass