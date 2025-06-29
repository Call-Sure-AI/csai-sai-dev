# src/application/services/conversation_service.py
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime

from core.interfaces.services import IConversationService, IAnalyticsService, IAgentService
from core.interfaces.repositories import IConversationRepository
from core.interfaces.external import IAIService
from core.entities.conversation import Conversation, ConversationStatus

logger = logging.getLogger(__name__)

class ConversationService(IConversationService):
    """Service for conversation management and message processing"""
    
    def __init__(
        self,
        conversation_repository: IConversationRepository,
        ai_service: IAIService,
        analytics_service: IAnalyticsService,
        agent_service: IAgentService
    ):
        self.conversation_repository = conversation_repository
        self.ai_service = ai_service
        self.analytics_service = analytics_service
        self.agent_service = agent_service
    
    async def process_message(
        self, 
        client_id: str, 
        message: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Process user message and return streaming AI response"""
        start_time = time.time()
        full_response = ""
        token_count = 0
        
        try:
            # Get conversation context
            # This would be injected from the client session in practice
            conversation_id = await self._get_conversation_id_for_client(client_id)
            context = await self.get_conversation_context(conversation_id)
            
            # Add user message to context
            context.append({"role": "user", "content": message})
            
            # Get agent prompt (would be from agent service)
            agent_prompt = await self._get_agent_prompt_for_client(client_id)
            
            # Generate AI response
            async for token in self.ai_service.generate_response(context, agent_prompt):
                full_response += token
                token_count += 1
                yield token
            
            # Record message in conversation
            conversation = await self.conversation_repository.get_conversation(conversation_id)
            if conversation:
                conversation.add_message("user", message)
                conversation.add_message("assistant", full_response, token_count, metadata)
                await self.conversation_repository.update_conversation(conversation)
            
            # Record analytics
            response_time = time.time() - start_time
            await self.analytics_service.record_message(client_id, token_count, response_time)
            
        except Exception as e:
            logger.error(f"Error processing message for {client_id}: {e}")
            # Record error
            company_id = await self._get_company_id_for_client(client_id)
            if company_id:
                await self.analytics_service.record_error(company_id, str(e))
            raise
    
    async def get_conversation_context(
        self, 
        conversation_id: str, 
        max_messages: int = 5
    ) -> List[Dict[str, str]]:
        """Get conversation context for AI"""
        conversation = await self.conversation_repository.get_conversation(conversation_id)
        if not conversation:
            return []
        
        return conversation.get_context_for_ai(max_messages)
    
    async def create_conversation(
        self, 
        customer_id: str, 
        company_id: str, 
        agent_id: Optional[str] = None
    ) -> str:
        """Create new conversation and return ID"""
        conversation = Conversation(
            id=f"conv_{int(time.time())}_{customer_id}",
            customer_id=customer_id,
            company_id=company_id,
            agent_id=agent_id
        )
        
        conversation_id = await self.conversation_repository.create_conversation(conversation)
        logger.info(f"Created conversation: {conversation_id}")
        return conversation_id
    
    async def end_conversation(self, conversation_id: str) -> None:
        """End conversation and update final metrics"""
        conversation = await self.conversation_repository.get_conversation(conversation_id)
        if not conversation:
            return
        
        conversation.status = ConversationStatus.ENDED
        await self.conversation_repository.update_conversation(conversation)
        
        logger.info(f"Ended conversation: {conversation_id}")
    
    # Helper methods - these would integrate with other services
    async def _get_conversation_id_for_client(self, client_id: str) -> str:
        """Get conversation ID for client (placeholder)"""
        # This would be retrieved from the client session
        return f"conv_{client_id}"
    
    async def _get_agent_prompt_for_client(self, client_id: str) -> Optional[str]:
        """Get agent prompt for client (placeholder)"""
        # This would be retrieved from the agent service
        return "You are a helpful AI assistant."
    
    async def _get_company_id_for_client(self, client_id: str) -> Optional[str]:
        """Get company ID for client (placeholder)"""
        # This would be retrieved from the client session
        return None

# src/application/services/voice_service.py
import asyncio
import logging
from typing import Optional, Callable
from datetime import datetime

from core.interfaces.services import IVoiceService, IAnalyticsService
from core.interfaces.external import ISpeechToTextService, ITextToSpeechService

logger = logging.getLogger(__name__)

class VoiceService(IVoiceService):
    """Service for voice call management"""
    
    def __init__(
        self,
        stt_service: ISpeechToTextService,
        tts_service: ITextToSpeechService,
        analytics_service: IAnalyticsService
    ):
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.analytics_service = analytics_service
        
        # Track active voice sessions
        self.active_sessions = {}
    
    async def start_voice_call(
        self, 
        client_id: str, 
        voice_callback: Optional[Callable] = None
    ) -> bool:
        """Start voice call session"""
        try:
            # Initialize STT session
            async def transcription_callback(session_id, text):
                if voice_callback:
                    await voice_callback(client_id, text)
            
            success = await self.stt_service.initialize_session(client_id, transcription_callback)
            if not success:
                return False
            
            # Track session
            self.active_sessions[client_id] = {
                "start_time": datetime.utcnow(),
                "callback": voice_callback
            }
            
            logger.info(f"Voice call started: {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting voice call for {client_id}: {e}")
            return False
    
    async def end_voice_call(self, client_id: str) -> float:
        """End voice call and return duration"""
        session = self.active_sessions.get(client_id)
        if not session:
            return 0.0
        
        try:
            # Calculate duration
            duration = (datetime.utcnow() - session["start_time"]).total_seconds()
            
            # Close STT session
            await self.stt_service.close_session(client_id)
            
            # Record analytics
            await self.analytics_service.record_voice_call(client_id, duration)
            
            # Remove session
            self.active_sessions.pop(client_id, None)
            
            logger.info(f"Voice call ended: {client_id}, duration: {duration:.1f}s")
            return duration
            
        except Exception as e:
            logger.error(f"Error ending voice call for {client_id}: {e}")
            return 0.0
    
    async def process_audio_chunk(self, client_id: str, audio_data: bytes) -> None:
        """Process incoming audio data"""
        if client_id not in self.active_sessions:
            logger.warning(f"Audio chunk received for inactive session: {client_id}")
            return
        
        try:
            await self.stt_service.process_audio_chunk(client_id, audio_data)
        except Exception as e:
            logger.error(f"Error processing audio for {client_id}: {e}")
    
    async def synthesize_speech(self, client_id: str, text: str) -> bytes:
        """Convert text to speech for client"""
        try:
            return await self.tts_service.generate_audio(text)
        except Exception as e:
            logger.error(f"Error synthesizing speech for {client_id}: {e}")
            return b""