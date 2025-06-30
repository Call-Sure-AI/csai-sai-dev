# src/infrastructure/database/repositories/conversation_repository.py
"""
Concrete implementation of conversation repository using SQLAlchemy.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import desc, and_

from core.interfaces.repositories import IConversationRepository
from core.entities.conversation import Conversation, Message, MessageType as CoreMessageType
from ..models import ConversationRecord, MessageRecord, MessageType

logger = logging.getLogger(__name__)

class ConversationRepository(IConversationRepository):
    """SQLAlchemy implementation of conversation repository."""
    
    def __init__(self, session: Session):
        self.session = session
    
    async def save_conversation(self, conversation: Conversation) -> None:
        """Save or update a conversation."""
        try:
            record = self.session.query(ConversationRecord).filter(
                ConversationRecord.id == conversation.id
            ).first()
            
            if record:
                # Update existing
                record.state = conversation.state.value
                record.context = conversation.context
                record.updated_at = conversation.updated_at
                record.message_count = conversation.metrics.total_messages
                record.total_tokens = conversation.metrics.total_tokens
                record.voice_duration = conversation.metrics.total_audio_duration
                
                if conversation.state.value in ["completed", "terminated"]:
                    record.ended_at = datetime.utcnow()
            else:
                # Create new
                record = ConversationRecord(
                    id=conversation.id,
                    client_id=conversation.client_id,
                    company_id="unknown",  # Would be set by service layer
                    agent_id=conversation.agent_id,
                    title=conversation.title,
                    state=conversation.state.value,
                    context=conversation.context,
                    created_at=conversation.created_at,
                    updated_at=conversation.updated_at,
                    message_count=conversation.metrics.total_messages,
                    total_tokens=conversation.metrics.total_tokens,
                    voice_duration=conversation.metrics.total_audio_duration
                )
                self.session.add(record)
            
            self.session.commit()
            
        except SQLAlchemyError as e:
            logger.error(f"Error saving conversation {conversation.id}: {e}")
            self.session.rollback()
            raise
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        try:
            record = self.session.query(ConversationRecord).filter(
                ConversationRecord.id == conversation_id
            ).first()
            
            if not record:
                return None
            
            # Convert to domain entity
            from core.entities.conversation import ConversationState
            
            conversation = Conversation(
                id=record.id,
                client_id=record.client_id,
                agent_id=record.agent_id,
                title=record.title,
                created_at=record.created_at,
                updated_at=record.updated_at
            )
            
            conversation.state = ConversationState(record.state)
            conversation.context = record.context or {}
            
            # Load messages
            messages = await self.get_messages(conversation_id)
            conversation.messages = messages
            
            return conversation
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting conversation {conversation_id}: {e}")
            return None
    
    async def get_conversations_by_client(self, client_id: str) -> List[Conversation]:
        """Get all conversations for a client."""
        try:
            records = self.session.query(ConversationRecord).filter(
                ConversationRecord.client_id == client_id
            ).order_by(desc(ConversationRecord.created_at)).all()
            
            conversations = []
            for record in records:
                conversation = await self.get_conversation(record.id)
                if conversation:
                    conversations.append(conversation)
            
            return conversations
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting conversations for client {client_id}: {e}")
            return []
    
    async def get_conversations_by_agent(self, agent_id: str) -> List[Conversation]:
        """Get all conversations handled by an agent."""
        try:
            records = self.session.query(ConversationRecord).filter(
                ConversationRecord.agent_id == agent_id
            ).order_by(desc(ConversationRecord.created_at)).all()
            
            conversations = []
            for record in records:
                conversation = await self.get_conversation(record.id)
                if conversation:
                    conversations.append(conversation)
            
            return conversations
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting conversations for agent {agent_id}: {e}")
            return []
    
    async def save_message(self, message: Message) -> None:
        """Save a message."""
        try:
            # Map core message type to database enum
            db_message_type = MessageType(message.message_type.value)
            
            record = MessageRecord(
                id=message.id,
                conversation_id=message.conversation_id,
                message_type=db_message_type,
                content=message.content,
                timestamp=message.timestamp,
                sender_id=message.sender_id,
                agent_id=message.agent_id,
                client_id=message.client_id,
                tokens_used=message.tokens_used,
                processing_time=message.processing_time,
                confidence_score=message.confidence_score,
                audio_duration=message.audio_duration,
                audio_format=message.audio_format,
                transcription_confidence=message.transcription_confidence,
                function_name=message.function_name,
                function_args=message.function_args,
                function_result=message.function_result,
                error_code=message.error_code,
                error_details=message.error_details,
                metadata=message.metadata
            )
            
            self.session.add(record)
            self.session.commit()
            
        except SQLAlchemyError as e:
            logger.error(f"Error saving message {message.id}: {e}")
            self.session.rollback()
            raise
    
    async def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Message]:
        """Get messages for a conversation."""
        try:
            query = self.session.query(MessageRecord).filter(
                MessageRecord.conversation_id == conversation_id
            ).order_by(MessageRecord.timestamp)
            
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            
            records = query.all()
            
            messages = []
            for record in records:
                # Convert to domain entity
                core_message_type = CoreMessageType(record.message_type.value)
                
                message = Message(
                    id=record.id,
                    conversation_id=record.conversation_id,
                    message_type=core_message_type,
                    content=record.content,
                    timestamp=record.timestamp,
                    sender_id=record.sender_id,
                    agent_id=record.agent_id,
                    client_id=record.client_id,
                    metadata=record.metadata or {},
                    tokens_used=record.tokens_used,
                    processing_time=record.processing_time,
                    confidence_score=record.confidence_score,
                    audio_duration=record.audio_duration,
                    audio_format=record.audio_format,
                    transcription_confidence=record.transcription_confidence,
                    function_name=record.function_name,
                    function_args=record.function_args,
                    function_result=record.function_result,
                    error_code=record.error_code,
                    error_details=record.error_details
                )
                messages.append(message)
            
            return messages
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting messages for conversation {conversation_id}: {e}")
            return []
    
    async def get_recent_conversations(
        self,
        limit: int = 10,
        company_id: Optional[str] = None
    ) -> List[Conversation]:
        """Get recent conversations."""
        try:
            query = self.session.query(ConversationRecord).order_by(
                desc(ConversationRecord.created_at)
            )
            
            if company_id:
                query = query.filter(ConversationRecord.company_id == company_id)
            
            records = query.limit(limit).all()
            
            conversations = []
            for record in records:
                conversation = await self.get_conversation(record.id)
                if conversation:
                    conversations.append(conversation)
            
            return conversations
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting recent conversations: {e}")
            return []
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages."""
        try:
            # Delete messages first (due to foreign key constraint)
            self.session.query(MessageRecord).filter(
                MessageRecord.conversation_id == conversation_id
            ).delete()
            
            # Delete conversation
            deleted = self.session.query(ConversationRecord).filter(
                ConversationRecord.id == conversation_id
            ).delete()
            
            self.session.commit()
            return deleted > 0
            
        except SQLAlchemyError as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            self.session.rollback()
            return False
    
    async def search_conversations(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Conversation]:
        """Search conversations by content or metadata."""
        # This would require full-text search capabilities
        # For now, return empty list as placeholder
        return []