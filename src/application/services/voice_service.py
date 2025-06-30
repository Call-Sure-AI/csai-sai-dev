# src/application/services/voice_service.py
"""
Voice service for managing voice calls and audio processing.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime

from core.interfaces.services import IVoiceService
from core.interfaces.external import ISpeechToTextService, ITextToSpeechService
from core.entities.client import ClientSession, VoiceCallState
from core.exceptions import VoiceCallException

logger = logging.getLogger(__name__)

class VoiceService(IVoiceService):
    """Service for voice call management and audio processing."""
    
    def __init__(
        self,
        stt_service: ISpeechToTextService,
        tts_service: ITextToSpeechService,
        max_call_duration: int = 3600  # 1 hour max
    ):
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.max_call_duration = max_call_duration
        
        # Track active voice sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def start_voice_call(
        self,
        client_id: str,
        session: ClientSession,
        transcription_callback: Optional[Callable] = None,
        voice_settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Start a voice call session."""
        try:
            if session.voice_call_state != VoiceCallState.INACTIVE:
                raise VoiceCallException(
                    operation="start",
                    reason=f"Voice call already in state: {session.voice_call_state}",
                    client_id=client_id
                )
            
            # Initialize voice call in session
            session.start_voice_call(transcription_callback)
            
            # Create voice session data
            voice_session = {
                "client_id": client_id,
                "start_time": datetime.utcnow(),
                "transcription_callback": transcription_callback,
                "settings": voice_settings or {},
                "audio_chunks_received": 0,
                "transcription_results": []
            }
            
            self.active_sessions[client_id] = voice_session
            
            # Activate voice call
            session.activate_voice_call()
            
            logger.info(f"Voice call started for client: {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting voice call for {client_id}: {e}")
            return False
    
    async def end_voice_call(self, client_id: str, session: ClientSession) -> float:
        """End a voice call and return duration."""
        try:
            if session.voice_call_state == VoiceCallState.INACTIVE:
                return 0.0
            
            # End voice call in session
            duration = session.end_voice_call()
            
            # Cleanup voice session data
            voice_session = self.active_sessions.pop(client_id, None)
            
            if voice_session:
                # Log voice call statistics
                stats = {
                    "client_id": client_id,
                    "duration_minutes": duration,
                    "audio_chunks_received": voice_session.get("audio_chunks_received", 0),
                    "transcription_results": len(voice_session.get("transcription_results", []))
                }
                logger.info(f"Voice call ended: {stats}")
            
            return duration
            
        except Exception as e:
            logger.error(f"Error ending voice call for {client_id}: {e}")
            return 0.0
    
    async def process_audio_chunk(
        self,
        client_id: str,
        audio_data: bytes,
        audio_format: str = "webm"
    ) -> Optional[Dict[str, Any]]:
        """Process incoming audio chunk for transcription."""
        voice_session = self.active_sessions.get(client_id)
        if not voice_session:
            logger.warning(f"Audio chunk received for inactive session: {client_id}")
            return None
        
        try:
            # Update chunk counter
            voice_session["audio_chunks_received"] += 1
            
            # Transcribe audio chunk
            transcription_result = await self.stt_service.transcribe_audio(
                audio_data=audio_data,
                audio_format=audio_format,
                language="en-US"
            )
            
            # Store result
            voice_session["transcription_results"].append(transcription_result)
            
            # Call transcription callback if available
            callback = voice_session.get("transcription_callback")
            if callback:
                try:
                    await callback(client_id, transcription_result)
                except Exception as e:
                    logger.error(f"Error in transcription callback: {e}")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for {client_id}: {e}")
            return None
    
    async def synthesize_speech(
        self,
        text: str,
        voice_settings: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Convert text to speech audio."""
        try:
            settings = voice_settings or {}
            
            audio_data = await self.tts_service.synthesize_speech(
                text=text,
                voice=settings.get("voice"),
                speed=settings.get("speed"),
                output_format=settings.get("format", "mp3")
            )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise VoiceCallException(
                operation="speech_synthesis",
                reason=str(e)
            )
    
    async def get_voice_session_stats(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get voice session statistics."""
        voice_session = self.active_sessions.get(client_id)
        if not voice_session:
            return None
        
        current_time = datetime.utcnow()
        duration = (current_time - voice_session["start_time"]).total_seconds()
        
        return {
            "client_id": client_id,
            "duration_seconds": duration,
            "audio_chunks_received": voice_session["audio_chunks_received"],
            "transcription_results_count": len(voice_session["transcription_results"]),
            "start_time": voice_session["start_time"].isoformat()
        }
    
    async def get_all_active_calls(self) -> List[Dict[str, Any]]:
        """Get statistics for all active voice calls."""
        stats = []
        for client_id in self.active_sessions:
            session_stats = await self.get_voice_session_stats(client_id)
            if session_stats:
                stats.append(session_stats)
        return stats
    
    async def cleanup_inactive_sessions(self) -> int:
        """Cleanup voice sessions that exceed max duration."""
        current_time = datetime.utcnow()
        inactive_sessions = []
        
        for client_id, voice_session in self.active_sessions.items():
            duration = (current_time - voice_session["start_time"]).total_seconds()
            if duration > self.max_call_duration:
                inactive_sessions.append(client_id)
        
        # Cleanup inactive sessions
        for client_id in inactive_sessions:
            self.active_sessions.pop(client_id, None)
            logger.warning(f"Cleaned up inactive voice session: {client_id}")
        
        return len(inactive_sessions)