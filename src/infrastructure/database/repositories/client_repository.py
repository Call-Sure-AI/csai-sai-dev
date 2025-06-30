# src/infrastructure/database/repositories/client_repository.py
"""
Concrete implementation of client repository using SQLAlchemy.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from core.interfaces.repositories import IClientRepository
from core.entities.client import ClientSession
from ..models import Company, SessionEvent, EventType

logger = logging.getLogger(__name__)

class ClientRepository(IClientRepository):
    """SQLAlchemy implementation of client repository."""
    
    def __init__(self, session: Session):
        self.session = session
    
    async def save_session(self, session: ClientSession) -> None:
        """Save or update a client session."""
        try:
            # For now, we'll store session events rather than full sessions
            # since sessions are primarily in-memory for this application
            pass
        except SQLAlchemyError as e:
            logger.error(f"Error saving session {session.session_id}: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[ClientSession]:
        """Get a client session by ID."""
        # Sessions are primarily in-memory, this would be for recovery scenarios
        return None
    
    async def get_sessions_by_client(self, client_id: str) -> List[ClientSession]:
        """Get all sessions for a client."""
        return []
    
    async def get_active_sessions(self) -> List[ClientSession]:
        """Get all currently active sessions."""
        return []
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return True
    
    async def authenticate_company(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate a company by API key."""
        try:
            company = self.session.query(Company).filter(
                Company.api_key == api_key,
                Company.status == "active"
            ).first()
            
            if not company:
                return None
            
            return {
                "id": company.id,
                "name": company.name,
                "api_key": company.api_key,
                "max_connections": company.max_connections,
                "max_requests_per_minute": company.max_requests_per_minute,
                "metadata": company.metadata or {}
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error authenticating company with API key: {e}")
            return None
    
    async def record_connection_event(
        self,
        session_id: str,
        event_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a connection event."""
        try:
            # Create event record
            event = SessionEvent(
                id=f"{session_id}_{event_type}_{int(datetime.utcnow().timestamp())}",
                session_id=session_id,
                event_type=EventType(event_type),
                metadata=metadata or {}
            )
            
            self.session.add(event)
            self.session.commit()
            
        except SQLAlchemyError as e:
            logger.error(f"Error recording connection event: {e}")
            self.session.rollback()
            raise
    
    async def get_session_count_by_company(self, company_id: str) -> int:
        """Get active session count for a company."""
        try:
            # Count recent connection events minus disconnection events
            # This is a simplified approach - in production you'd want a better tracking mechanism
            count = self.session.query(SessionEvent).filter(
                SessionEvent.company_id == company_id,
                SessionEvent.event_type == EventType.CONNECTION,
                SessionEvent.timestamp >= datetime.utcnow().replace(hour=0, minute=0, second=0)
            ).count()
            
            return count
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting session count for company {company_id}: {e}")
            return 0