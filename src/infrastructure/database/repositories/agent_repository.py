# src/infrastructure/database/repositories/agent_repository.py
import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from core.interfaces.repositories import IAgentRepository
from database.models import Agent, Company

logger = logging.getLogger(__name__)

class AgentRepository(IAgentRepository):
    """Repository for agent data persistence"""
    
    def __init__(self, session: Session):
        self.session = session
    
    async def get_agent(self, agent_id: str, company_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information"""
        try:
            agent = self.session.query(Agent).filter_by(
                id=agent_id,
                company_id=company_id,
                is_active=True
            ).first()
            
            if not agent:
                return None
            
            return {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type,
                "prompt": agent.prompt,
                "confidence_threshold": agent.confidence_threshold,
                "additional_context": agent.additional_context or {},
                "advanced_settings": agent.advanced_settings or {},
                "files": agent.files or [],
                "created_at": agent.created_at,
                "updated_at": agent.updated_at
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting agent: {e}")
            raise
    
    async def get_company_agents(self, company_id: str) -> List[Dict[str, Any]]:
        """Get all agents for company"""
        try:
            agents = self.session.query(Agent).filter_by(
                company_id=company_id,
                is_active=True
            ).all()
            
            return [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "type": agent.type,
                    "prompt": agent.prompt,
                    "confidence_threshold": agent.confidence_threshold,
                    "additional_context": agent.additional_context or {},
                    "document_count": len(agent.documents) if agent.documents else 0
                }
                for agent in agents
            ]
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting company agents: {e}")
            raise
    
    async def get_base_agent(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Get base agent for company"""
        try:
            agent = self.session.query(Agent).filter_by(
                company_id=company_id,
                type="base",
                is_active=True
            ).first()
            
            if not agent:
                return None
            
            return {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type,
                "prompt": agent.prompt,
                "confidence_threshold": agent.confidence_threshold,
                "additional_context": agent.additional_context or {}
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting base agent: {e}")
            raise