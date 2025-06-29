import logging
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from core.interfaces.repositories import IConversationRepository
from core.entities.conversation import Conversation, ConversationStatus, Message
from database.models import Conversation as ConversationModel
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ConversationRepository(IConversationRepository):
    """Repository for conversation persistence"""
    
    def __init__(self, session: Session):
        self.session = session
    
    async def create_conversation(self, conversation: Conversation) -> str:
        """Create new conversation and return ID"""
        try:
            db_conversation = ConversationModel(
                id=conversation.id,
                customer_id=conversation.customer_id,
                company_id=conversation.company_id,
                current_agent_id=conversation.agent_id,
                history=[],
                meta_data=conversation.context,
                status="active",
                created_at=conversation.created_at,
                updated_at=conversation.updated_at
            )
            
            self.session.add(db_conversation)
            self.session.commit()
            
            logger.info(f"Created conversation: {conversation.id}")
            return conversation.id
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating conversation: {e}")
            self.session.rollback()
            raise
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        try:
            db_conversation = self.session.query(ConversationModel).filter_by(
                id=conversation_id
            ).first()
            
            if not db_conversation:
                return None
            
            # Convert database model to domain entity
            conversation = Conversation(
                id=db_conversation.id,
                customer_id=db_conversation.customer_id,
                company_id=db_conversation.company_id,
                agent_id=db_conversation.current_agent_id,
                status=ConversationStatus(db_conversation.status),
                context=db_conversation.meta_data or {},
                created_at=db_conversation.created_at,
                updated_at=db_conversation.updated_at
            )
            
            # Convert history to messages
            history = db_conversation.history or []
            for msg_data in history:
                if isinstance(msg_data, dict):
                    message = Message(
                        role=msg_data.get("role", "user"),
                        content=msg_data.get("content", ""),
                        timestamp=datetime.fromisoformat(msg_data.get("timestamp", datetime.utcnow().isoformat())),
                        metadata=msg_data.get("metadata", {}),
                        tokens=msg_data.get("tokens", 0)
                    )
                    conversation.messages.append(message)
            
            return conversation
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting conversation: {e}")
            raise
    
    async def update_conversation(self, conversation: Conversation) -> None:
        """Update existing conversation"""
        try:
            db_conversation = self.session.query(ConversationModel).filter_by(
                id=conversation.id
            ).first()
            
            if not db_conversation:
                logger.warning(f"Conversation not found for update: {conversation.id}")
                return
            
            # Convert messages to history format
            history = []
            for message in conversation.messages:
                history.append({
                    "role": message.role,
                    "content": message.content,
                    "timestamp": message.timestamp.isoformat(),
                    "metadata": message.metadata,
                    "tokens": message.tokens
                })
            
            # Update fields
            db_conversation.history = history
            db_conversation.current_agent_id = conversation.agent_id
            db_conversation.status = conversation.status.value
            db_conversation.meta_data = conversation.context
            db_conversation.updated_at = conversation.updated_at
            db_conversation.messages_count = conversation.message_count
            
            self.session.commit()
            logger.debug(f"Updated conversation: {conversation.id}")
            
        except SQLAlchemyError as e:
            logger.error(f"Error updating conversation: {e}")
            self.session.rollback()
            raise
    
    async def get_or_create_conversation(
        self, 
        customer_id: str, 
        company_id: str, 
        agent_id: Optional[str] = None
    ) -> Conversation:
        """Get existing or create new conversation"""
        try:
            # Try to find existing active conversation
            db_conversation = self.session.query(ConversationModel).filter_by(
                customer_id=customer_id,
                company_id=company_id,
                status="active"
            ).order_by(ConversationModel.created_at.desc()).first()
            
            if db_conversation:
                # Return existing conversation
                return await self.get_conversation(db_conversation.id)
            
            # Create new conversation
            conversation = Conversation(
                id=f"conv_{customer_id}_{int(datetime.utcnow().timestamp())}",
                customer_id=customer_id,
                company_id=company_id,
                agent_id=agent_id
            )
            
            await self.create_conversation(conversation)
            return conversation
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting or creating conversation: {e}")
            raise