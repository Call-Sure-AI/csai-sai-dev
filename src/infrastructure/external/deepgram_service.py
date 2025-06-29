# src/infrastructure/external/deepgram_service.py
import asyncio
import websockets
import json
import logging
from typing import Callable, Dict, Any, Optional

from core.interfaces.external import ISpeechToTextService

logger = logging.getLogger(__name__)

class DeepgramSTTService(ISpeechToTextService):
    """Deepgram speech-to-text service implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "wss://api.deepgram.com/v1/listen"
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def initialize_session(self, session_id: str, callback: Callable) -> bool:
        """Initialize STT session with callback"""
        try:
            # Build connection parameters
            params = self._build_connection_params()
            headers = {"Authorization": f"Token {self.api_key}"}
            
            # Establish WebSocket connection
            websocket = await websockets.connect(
                f"{self.base_url}?{params}",
                extra_headers=headers
            )
            
            # Store session
            self.active_sessions[session_id] = {
                "websocket": websocket,
                "callback": callback,
                "active": True
            }
            
            # Start listening for transcriptions
            asyncio.create_task(self._listen_for_transcriptions(session_id))
            
            logger.info(f"Initialized Deepgram session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Deepgram session {session_id}: {e}")
            return False
    
    async def process_audio_chunk(self, session_id: str, audio_data: bytes) -> None:
        """Process audio chunk"""
        session = self.active_sessions.get(session_id)
        if not session or not session["active"]:
            return
        
        try:
            websocket = session["websocket"]
            await websocket.send(audio_data)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for {session_id}: {e}")
            await self.close_session(session_id)
    
    async def close_session(self, session_id: str) -> None:
        """Close STT session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        try:
            session["active"] = False
            websocket = session["websocket"]
            
            if not websocket.closed:
                await websocket.close()
            
            self.active_sessions.pop(session_id, None)
            logger.info(f"Closed Deepgram session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error closing Deepgram session {session_id}: {e}")
    
    async def _listen_for_transcriptions(self, session_id: str) -> None:
        """Listen for transcription results"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        websocket = session["websocket"]
        callback = session["callback"]
        
        try:
            async for message in websocket:
                if not session["active"]:
                    break
                
                try:
                    response = json.loads(message)
                    
                    # Extract transcript
                    transcript = self._extract_transcript(response)
                    if transcript:
                        await callback(session_id, transcript)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing Deepgram response: {e}")
                    
        except Exception as e:
            logger.error(f"Error listening for transcriptions {session_id}: {e}")
        finally:
            await self.close_session(session_id)
    
    def _build_connection_params(self) -> str:
        """Build connection parameters for Deepgram"""
        params = {
            "model": "nova",
            "language": "en-US",
            "encoding": "linear16",
            "sample_rate": "16000",
            "channels": "1",
            "interim_results": "true",
            "punctuate": "true",
            "diarize": "false"
        }
        
        return "&".join([f"{k}={v}" for k, v in params.items()])
    
    def _extract_transcript(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract transcript from Deepgram response"""
        try:
            if "channel" in response:
                alternatives = response["channel"]["alternatives"]
                if alternatives and len(alternatives) > 0:
                    return alternatives[0]["transcript"]
            return None
        except (KeyError, IndexError):
            return None