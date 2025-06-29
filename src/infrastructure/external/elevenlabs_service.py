# src/infrastructure/external/elevenlabs_service.py
import httpx
import logging
from typing import Optional, Dict, Any, Callable

from core.interfaces.external import ITextToSpeechService

logger = logging.getLogger(__name__)

class ElevenLabsTTSService(ITextToSpeechService):
    """ElevenLabs text-to-speech service implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        self.default_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
    
    async def generate_audio(self, text: str, voice_settings: Optional[Dict] = None) -> bytes:
        """Generate audio from text"""
        try:
            voice_id = voice_settings.get("voice_id", self.default_voice_id) if voice_settings else self.default_voice_id
            
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            payload = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": voice_settings.get("stability", 0.5) if voice_settings else 0.5,
                    "similarity_boost": voice_settings.get("similarity_boost", 0.5) if voice_settings else 0.5
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                return response.content
                
        except Exception as e:
            logger.error(f"Error generating ElevenLabs audio: {e}")
            return b""
    
    async def stream_audio(
        self, 
        text: str, 
        callback: Callable,
        voice_settings: Optional[Dict] = None
    ) -> None:
        """Stream audio generation"""
        try:
            voice_id = voice_settings.get("voice_id", self.default_voice_id) if voice_settings else self.default_voice_id
            
            url = f"{self.base_url}/text-to-speech/{voice_id}/stream"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            payload = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": voice_settings.get("stability", 0.5) if voice_settings else 0.5,
                    "similarity_boost": voice_settings.get("similarity_boost", 0.5) if voice_settings else 0.5
                }
            }
            
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            await callback(chunk)
                            
        except Exception as e:
            logger.error(f"Error streaming ElevenLabs audio: {e}")