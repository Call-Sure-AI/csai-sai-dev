# src/infrastructure/external/deepgram_service.py
"""
Deepgram service implementation for speech-to-text operations.
"""

import logging
import asyncio
import json
import websockets
from typing import Optional, List, Dict, Any, AsyncGenerator
import aiohttp

from core.interfaces.external import ISpeechToTextService

logger = logging.getLogger(__name__)

class DeepgramService(ISpeechToTextService):
    """Deepgram implementation of speech-to-text service."""
    
    def __init__(self, api_key: str, default_model: str = "nova-2"):
        self.api_key = api_key
        self.default_model = default_model
        self.base_url = "https://api.deepgram.com/v1"
        self.ws_url = "wss://api.deepgram.com/v1/listen"
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        language: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe audio to text."""
        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": f"audio/{audio_format}"
            }
            
            params = {
                "model": model or self.default_model,
                "punctuate": kwargs.get("punctuate", True),
                "diarize": kwargs.get("diarize", False),
                "detect_language": kwargs.get("detect_language", False),
            }
            
            if language:
                params["language"] = language
            
            url = f"{self.base_url}/listen"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    params=params,
                    data=audio_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract transcript and metadata
                        transcript = ""
                        confidence = 0.0
                        
                        if result.get("results") and result["results"].get("channels"):
                            channel = result["results"]["channels"][0]
                            if channel.get("alternatives"):
                                alternative = channel["alternatives"][0]
                                transcript = alternative.get("transcript", "")
                                confidence = alternative.get("confidence", 0.0)
                        
                        return {
                            "transcript": transcript,
                            "confidence": confidence,
                            "language": result.get("results", {}).get("language"),
                            "duration": result.get("metadata", {}).get("duration"),
                            "raw_response": result
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Deepgram API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    async def transcribe_audio_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        audio_format: str = "wav",
        language: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Transcribe audio stream in real-time."""
        try:
            # Build WebSocket URL with parameters
            params = {
                "model": model or self.default_model,
                "punctuate": kwargs.get("punctuate", True),
                "interim_results": kwargs.get("interim_results", True),
                "endpointing": kwargs.get("endpointing", True),
                "encoding": audio_format,
                "sample_rate": kwargs.get("sample_rate", 16000),
                "channels": kwargs.get("channels", 1)
            }
            
            if language:
                params["language"] = language
            
            # Build query string
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            ws_url_with_params = f"{self.ws_url}?{query_string}"
            
            headers = {"Authorization": f"Token {self.api_key}"}
            
            async with websockets.connect(
                ws_url_with_params,
                extra_headers=headers
            ) as websocket:
                
                # Start tasks for sending audio and receiving transcripts
                async def send_audio():
                    try:
                        async for audio_chunk in audio_stream:
                            await websocket.send(audio_chunk)
                        # Send close message
                        await websocket.send(json.dumps({"type": "CloseStream"}))
                    except Exception as e:
                        logger.error(f"Error sending audio: {e}")
                
                async def receive_transcripts():
                    try:
                        async for message in websocket:
                            data = json.loads(message)
                            
                            if data.get("type") == "Results":
                                channel = data.get("channel", {})
                                alternatives = channel.get("alternatives", [])
                                
                                if alternatives:
                                    alternative = alternatives[0]
                                    yield {
                                        "transcript": alternative.get("transcript", ""),
                                        "confidence": alternative.get("confidence", 0.0),
                                        "is_final": data.get("is_final", False),
                                        "speech_final": data.get("speech_final", False),
                                        "duration": data.get("duration"),
                                        "start": data.get("start")
                                    }
                                    
                    except Exception as e:
                        logger.error(f"Error receiving transcripts: {e}")
                
                # Run both tasks concurrently
                send_task = asyncio.create_task(send_audio())
                
                async for result in receive_transcripts():
                    yield result
                
                # Wait for send task to complete
                await send_task
                
        except Exception as e:
            logger.error(f"Error in audio stream transcription: {e}")
            raise
    
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # Deepgram supported languages (subset)
        return [
            "en", "en-US", "en-GB", "en-AU", "en-NZ", "en-IN",
            "es", "es-ES", "es-419",
            "fr", "fr-FR", "fr-CA",
            "de", "de-DE",
            "it", "it-IT",
            "pt", "pt-BR", "pt-PT",
            "ru", "ru-RU",
            "zh", "zh-CN", "zh-TW",
            "ja", "ja-JP",
            "ko", "ko-KR",
            "hi", "hi-IN",
            "ar", "ar-SA"
        ]
    
    async def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return [
            "wav", "mp3", "m4a", "flac", "opus", "ogg", "webm",
            "mp4", "mpeg", "mpga", "oga", "wav"
        ]
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available transcription models."""
        return [
            {"name": "nova-2", "description": "Latest and most accurate model"},
            {"name": "nova", "description": "Previous generation model"},
            {"name": "enhanced", "description": "Enhanced model for better accuracy"},
            {"name": "base", "description": "Base model for general use"}
        ]