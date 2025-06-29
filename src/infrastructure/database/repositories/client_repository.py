# src/infrastructure/database/repositories/client_repository.py
import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from core.interfaces.repositories import IClientRepository
from core.entities.client import ClientSession
from core.entities.analytics import SessionEvent, EventType
from database.models import Company, SessionEvent as SessionEventModel
from datetime import datetime

logger = logging.getLogger(__name__)

class ClientRepository(IClientRepository):
    """Repository for client data persistence"""
    
    def __init__(self, session: Session):
        self.session = session
    
    async def record_connection(self, session: ClientSession) -> None:
        """Record a new client connection"""
        if not session.company:
            return
        
        try:
            event = SessionEventModel(
                id=f"{session.client_id}_connect_{int(datetime.utcnow().timestamp())}",
                company_id=session.company["id"],
                client_id=session.client_id,
                event_type="connect",
                timestamp=session.connection_time
            )
            
            self.session.add(event)
            self.session.commit()
            logger.debug(f"Recorded connection for client: {session.client_id}")
            
        except SQLAlchemyError as e:
            logger.error(f"Error recording connection: {e}")
            self.session.rollback()
            raise
    
    async def record_disconnection(self, session: ClientSession) -> None:
        """Record client disconnection with session data"""
        if not session.company:
            return
        
        try:
            event = SessionEventModel(
                id=f"{session.client_id}_disconnect_{int(datetime.utcnow().timestamp())}",
                company_id=session.company["id"],
                client_id=session.client_id,
                event_type="disconnect",
                session_duration=session.get_session_duration(),
                message_count=session.message_count,
                token_count=session.total_tokens,
                timestamp=datetime.utcnow()
            )
            
            self.session.add(event)
            self.session.commit()
            logger.debug(f"Recorded disconnection for client: {session.client_id}")
            
        except SQLAlchemyError as e:
            logger.error(f"Error recording disconnection: {e}")
            self.session.rollback()
            raise
    
    async def authenticate_company(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate company by API key"""
        try:
            company = self.session.query(Company).filter_by(
                api_key=api_key,
                active=True
            ).first()
            
            if not company:
                logger.warning(f"Authentication failed for API key: {api_key[:8]}...")
                return None
            
            return {
                "id": company.id,
                "name": company.name,
                "api_key": company.api_key,
                "settings": company.settings or {}
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error authenticating company: {e}")
            raise
    
    async def get_company_by_id(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Get company information by ID"""
        try:
            company = self.session.query(Company).filter_by(
                id=company_id,
                active=True
            ).first()
            
            if not company:
                return None
            
            return {
                "id": company.id,
                "name": company.name,
                "api_key": company.api_key,
                "settings": company.settings or {},
                "created_at": company.created_at,
                "updated_at": company.updated_at
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting company: {e}")
            raise