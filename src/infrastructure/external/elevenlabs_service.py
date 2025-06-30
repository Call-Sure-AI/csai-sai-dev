# src/infrastructure/external/elevenlabs_service.py
"""
ElevenLabs service implementation for text-to-speech operations.
"""

import logging
import aiohttp
from typing import Optional, List, Dict, Any, AsyncGenerator
import json

from core.interfaces.external import ITextToSpeechService

logger = logging.getLogger(__name__)

class ElevenLabsService(ITextToSpeechService):
    """ElevenLabs implementation of text-to-speech service."""
    
    def __init__(self, api_key: str, default_voice: str = "21m00Tcm4TlvDq8ikWAM"):
        self.api_key = api_key
        self.default_voice = default_voice
        self.base_url = "https://api.elevenlabs.io/v1"
    
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
        try:
            voice_id = voice or self.default_voice
            
            headers = {
                "Accept": f"audio/{output_format}",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            # Voice settings
            voice_settings = {
                "stability": kwargs.get("stability", 0.75),
                "similarity_boost": kwargs.get("similarity_boost", 0.75),
                "style": kwargs.get("style", 0.0),
                "use_speaker_boost": kwargs.get("use_speaker_boost", True)
            }
            
            if speed is not None:
                voice_settings["speed"] = speed
            
            data = {
                "text": text,
                "model_id": kwargs.get("model_id", "eleven_monolingual_v1"),
                "voice_settings": voice_settings
            }
            
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        error_text = await response.text()
                        raise Exception(f"ElevenLabs API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise
    
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
        try:
            voice_id = voice or self.default_voice
            
            headers = {
                "Accept": f"audio/{output_format}",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            voice_settings = {
                "stability": kwargs.get("stability", 0.75),
                "similarity_boost": kwargs.get("similarity_boost", 0.75),
                "style": kwargs.get("style", 0.0),
                "use_speaker_boost": kwargs.get("use_speaker_boost", True)
            }
            
            if speed is not None:
                voice_settings["speed"] = speed
            
            data = {
                "text": text,
                "model_id": kwargs.get("model_id", "eleven_monolingual_v1"),
                "voice_settings": voice_settings
            }
            
            url = f"{self.base_url}/text-to-speech/{voice_id}/stream"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        async for chunk in response.content.iter_chunked(8192):
                            yield chunk
                    else:
                        error_text = await response.text()
                        raise Exception(f"ElevenLabs API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Error streaming speech synthesis: {e}")
            raise
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices."""
        try:
            headers = {"xi-api-key": self.api_key}
            url = f"{self.base_url}/voices"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("voices", [])
                    else:
                        error_text = await response.text()
                        raise Exception(f"ElevenLabs API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []
    
    async def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """Get information about a specific voice."""
        try:
            headers = {"xi-api-key": self.api_key}
            url = f"{self.base_url}/voices/{voice_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"ElevenLabs API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Error getting voice info for {voice_id}: {e}")
            return {}
    
    async def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return ["mp3", "wav", "ogg", "aac", "flac"]
    
    async def clone_voice(
        self,
        name: str,
        audio_samples: List[bytes],
        description: Optional[str] = None
    ) -> str:
        """Clone a voice from audio samples."""
        try:
            headers = {"xi-api-key": self.api_key}
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field("name", name)
            if description:
                data.add_field("description", description)
            
            # Add audio samples
            for i, sample in enumerate(audio_samples):
                data.add_field(
                    "files",
                    sample,
                    filename=f"sample_{i}.wav",
                    content_type="audio/wav"
                )
            
            url = f"{self.base_url}/voices/add"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    data=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("voice_id", "")
                    else:
                        error_text = await response.text()
                        raise Exception(f"ElevenLabs API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Error cloning voice: {e}")
            raise