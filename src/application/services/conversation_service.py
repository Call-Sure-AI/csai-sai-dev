# src/application/services/conversation_service.py
"""
Conversation service for managing AI conversations and message processing.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime

from core.interfaces.services import IConversationService
from core.interfaces.repositories import IConversationRepository
from core.interfaces.external import ILanguageModelService
from core.entities.conversation import Conversation, ConversationState, MessageType
from core.exceptions import ConversationNotFoundException, InvalidMessageFormatException

logger = logging.getLogger(__name__)

class ConversationService(IConversationService):
    """Service for conversation management and AI interactions."""
    
    def __init__(
        self,
        conversation_repository: IConversationRepository,
        language_model_service: ILanguageModelService,
        default_system_prompt: str = "You are a helpful AI assistant.",
        max_context_length: int = 4000
    ):
        self.conversation_repository = conversation_repository
        self.language_model_service = language_model_service
        self.default_system_prompt = default_system_prompt
        self.max_context_length = max_context_length
    
    async def create_conversation(
        self,
        client_id: str,
        agent_id: Optional[str] = None,
        title: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new conversation."""
        try:
            conversation_id = f"conv_{int(time.time())}_{client_id}"
            
            conversation = Conversation(
                id=conversation_id,
                client_id=client_id,
                agent_id=agent_id,
                title=title or f"Conversation with {client_id}"
            )
            
            if context:
                conversation.context.update(context)
            
            conversation.activate()
            
            # Save to repository
            await self.conversation_repository.save_conversation(conversation)
            
            logger.info(f"Created conversation: {conversation_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise
    
    async def process_message(
        self,
        conversation_id: str,
        message_content: str,
        client_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process user message and generate AI response."""
        start_time = time.time()
        full_response = ""
        token_count = 0
        
        try:
            # Get conversation
            conversation = await self.conversation_repository.get_conversation(conversation_id)
            if not conversation:
                raise ConversationNotFoundException(conversation_id)
            
            # Validate message
            if not message_content.strip():
                raise InvalidMessageFormatException("text", "Message content cannot be empty")
            
            # Add user message
            user_message = conversation.add_user_text_message(
                content=message_content,
                metadata=metadata or {}
            )
            
            # Save user message
            await self.conversation_repository.save_message(user_message)
            
            # Prepare context for AI
            context_messages = await self._prepare_ai_context(conversation)
            
            # Generate AI response
            response_start = time.time()
            async for chunk in self.language_model_service.generate_chat_stream(
                messages=context_messages,
                temperature=0.7,
                max_tokens=500
            ):
                if chunk.get("type") == "content" and chunk.get("content"):
                    content = chunk["content"]
                    full_response += content
                    token_count += 1
                    
                    yield {
                        "type": "stream_chunk",
                        "content": content,
                        "conversation_id": conversation_id,
                        "message_id": user_message.id,
                        "is_final": False
                    }
            
            # Calculate processing time
            processing_time = time.time() - response_start
            
            # Add agent response to conversation
            agent_message = conversation.add_agent_response(
                content=full_response,
                tokens_used=token_count,
                processing_time=processing_time,
                metadata={"model": "gpt-4", "temperature": 0.7}
            )
            
            # Save agent message and conversation
            await self.conversation_repository.save_message(agent_message)
            await self.conversation_repository.save_conversation(conversation)
            
            # Send final chunk
            yield {
                "type": "stream_complete",
                "content": full_response,
                "conversation_id": conversation_id,
                "message_id": agent_message.id,
                "tokens_used": token_count,
                "processing_time": processing_time,
                "is_final": True
            }
            
            total_time = time.time() - start_time
            logger.info(f"Processed message for conversation {conversation_id} in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing message for conversation {conversation_id}: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "conversation_id": conversation_id,
                "is_final": True
            }
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        return await self.conversation_repository.get_conversation(conversation_id)
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation message history."""
        try:
            messages = await self.conversation_repository.get_messages(
                conversation_id=conversation_id,
                limit=limit
            )
            
            return [message.to_dict() for message in messages]
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    async def end_conversation(self, conversation_id: str) -> bool:
        """End a conversation."""
        try:
            conversation = await self.conversation_repository.get_conversation(conversation_id)
            if not conversation:
                return False
            
            conversation.complete()
            await self.conversation_repository.save_conversation(conversation)
            
            logger.info(f"Ended conversation: {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error ending conversation {conversation_id}: {e}")
            return False
    
    async def update_conversation_context(
        self,
        conversation_id: str,
        context_updates: Dict[str, Any]
    ) -> bool:
        """Update conversation context."""
        try:
            conversation = await self.conversation_repository.get_conversation(conversation_id)
            if not conversation:
                return False
            
            for key, value in context_updates.items():
                conversation.update_context(key, value)
            
            await self.conversation_repository.save_conversation(conversation)
            return True
            
        except Exception as e:
            logger.error(f"Error updating conversation context: {e}")
            return False
    
    async def search_conversations(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search conversations."""
        try:
            conversations = await self.conversation_repository.search_conversations(
                query=query,
                filters=filters
            )
            
            return [conv.to_dict() for conv in conversations]
            
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []
    
    async def _prepare_ai_context(self, conversation: Conversation) -> List[Dict[str, str]]:
        """Prepare context messages for AI model."""
        messages = []
        
        # Add system prompt
        system_prompt = conversation.get_context("system_prompt", self.default_system_prompt)
        messages.append({"role": "system", "content": system_prompt})
        
        # Get conversation context window
        context_messages = conversation.get_context_window(self.max_context_length)
        
        # Convert to AI format
        for message in context_messages:
            if message.message_type == MessageType.USER_TEXT:
                messages.append({"role": "user", "content": message.content})
            elif message.message_type == MessageType.AGENT_TEXT:
                messages.append({"role": "assistant", "content": message.content})
            elif message.message_type == MessageType.SYSTEM:
                messages.append({"role": "system", "content": message.content})
        
        return messages