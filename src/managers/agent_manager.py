from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session
import logging
from datetime import datetime
import uuid
import asyncio
from database.models import Agent, Company, Document, Conversation, AgentInteraction
from config.settings import settings
import time

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self, db_session: Session, vector_store):
        self.db = db_session
        self.vector_store = vector_store
        self.agent_cache = {}
        self.company_agents_cache = {}
        self.conversation_cache = {}
        self._initialization_locks = {}

    # async def ensure_base_agent(self, company_id: str) -> Optional[Dict[str, Any]]:
    #     """Ensure base agent exists for company"""
    #     try:
    #         base_agent = await self.get_base_agent(company_id)
    #         if base_agent:
    #             return base_agent

    #         # Create default base agent
    #         agent_id = str(uuid.uuid4())
    #         base_agent = Agent(
    #             id=agent_id,
    #             company_id=company_id,
    #             name="Base Agent",
    #             type="base",
    #             prompt="You are a helpful AI assistant.",
    #             confidence_threshold=0.0,
    #             active=True
    #         )

    #         self.db.add(base_agent)
    #         self.db.commit()

    #         agent_info = {
    #             "id": agent_id,
    #             "name": base_agent.name,
    #             "type": base_agent.type,
    #             "prompt": base_agent.prompt,
    #             "confidence_threshold": base_agent.confidence_threshold
    #         }

    #         self.agent_cache[agent_id] = agent_info
    #         return agent_info

    #     except Exception as e:
    #         logger.error(f"Error ensuring base agent: {str(e)}")
    #         return None

    async def ensure_base_agent(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Ensure base agent exists for company"""
        try:
            base_agent = await self.get_base_agent(company_id)
            if base_agent:
                return base_agent

            # Get company to get the user_id
            company = self.db.query(Company).filter_by(id=company_id).first()
            if not company:
                logger.error(f"Company {company_id} not found")
                return None
                
            # Get user_id from company
            user_id = company.user_id
            if not user_id:
                logger.error(f"No user_id found for company {company_id}")
                return None

            # Create default base agent with user_id from company
            agent_id = str(uuid.uuid4())
            base_agent = Agent(
                id=agent_id,
                company_id=company_id,
                user_id=user_id,  # Use user_id from company
                name="Base Agent",
                type="base",
                prompt="You are a helpful AI assistant.",
                confidence_threshold=0.0,
                is_active=True  # Changed from active to is_active
            )

            self.db.add(base_agent)
            self.db.commit()

            agent_info = {
                "id": agent_id,
                "name": base_agent.name,
                "type": base_agent.type,
                "prompt": base_agent.prompt,
                "confidence_threshold": base_agent.confidence_threshold
            }

            self.agent_cache[agent_id] = agent_info
            return agent_info

        except Exception as e:
            logger.error(f"Error ensuring base agent: {str(e)}")
            self.db.rollback()  # Add rollback on exception
            return None
        
    async def initialize_company_agents(self, company_id: str) -> None:
        """Initialize agents with locking and caching"""
        lock = self._initialization_locks.get(company_id)
        if not lock:
            lock = asyncio.Lock()
            self._initialization_locks[company_id] = lock

        async with lock:
            try:
                if company_id in self.company_agents_cache:
                    return

                # Ensure base agent exists
                base_agent = await self.ensure_base_agent(company_id)
                if not base_agent:
                    raise ValueError("Failed to create base agent")

                # Get all active agents
                agents = self.db.query(Agent).filter_by(
                    company_id=company_id,
                    is_active=True  # Changed from active to is_active
                ).all()

                self.company_agents_cache[company_id] = []

                for agent in agents:
                    agent_info = {
                        "id": agent.id,
                        "name": agent.name,
                        "type": agent.type,
                        "prompt": agent.prompt,
                        "confidence_threshold": agent.confidence_threshold,
                        "additional_context": agent.additional_context
                    }
                    
                    self.agent_cache[agent.id] = agent_info
                    self.company_agents_cache[company_id].append(agent.id)

                    # Load documents if any exist
                    documents = self.db.query(Document).filter_by(agent_id=agent.id).all()
                    if documents:
                        docs_data = [{
                            'id': doc.id,
                            'content': doc.content,
                            'metadata': {
                                'agent_id': agent.id,
                                'file_type': doc.file_type,
                                'original_filename': doc.original_filename,
                                'doc_type': doc.type
                            }
                        } for doc in documents]
                        
                        await self.vector_store.load_documents(
                            company_id=company_id,
                            agent_id=agent.id,
                            documents=docs_data
                        )

                logger.info(f"Initialized {len(agents)} agents for company {company_id}")

            except Exception as e:
                logger.error(f"Error initializing agents: {str(e)}")
                self.agent_cache.pop(company_id, None)
                self.company_agents_cache.pop(company_id, None)
                raise
            
    async def get_company_agents(self, company_id: str) -> List[Dict[str, Any]]:
        """Get all agents for company"""
        if company_id not in self.company_agents_cache:
            await self.initialize_company_agents(company_id)

        return [
            self.agent_cache[agent_id]
            for agent_id in self.company_agents_cache.get(company_id, [])
        ]

    async def get_base_agent(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Get base agent with caching"""
        try:
            # Check cache first
            if company_id in self.company_agents_cache:
                for agent_id in self.company_agents_cache[company_id]:
                    agent = self.agent_cache.get(agent_id)
                    if agent and agent['type'] == 'base':
                        return agent

            # Query database
            base_agent = self.db.query(Agent).filter_by(
                company_id=company_id,
                type='base',
                is_active=True
            ).first()

            if base_agent:
                agent_info = {
                    "id": base_agent.id,
                    "name": base_agent.name,
                    "type": base_agent.type,
                    "prompt": base_agent.prompt,
                    "confidence_threshold": base_agent.confidence_threshold,
                    "additional_context": base_agent.additional_context
                }
                
                self.agent_cache[base_agent.id] = agent_info
                if company_id not in self.company_agents_cache:
                    self.company_agents_cache[company_id] = []
                self.company_agents_cache[company_id].append(base_agent.id)
                
                return agent_info

            return None

        except Exception as e:
            logger.error(f"Error getting base agent: {str(e)}")
            return None

    async def create_conversation(
        self,
        company_id: str,
        client_id: str
    ) -> Optional[Dict[str, Any]]:
        """Create conversation with proper error handling"""
        try:
            # Ensure base agent exists
            base_agent = await self.ensure_base_agent(company_id)
            if not base_agent:
                raise ValueError("No base agent available")

            conversation_id = str(uuid.uuid4())
            conversation = Conversation(
                id=conversation_id,
                customer_id=client_id,
                company_id=company_id,
                current_agent_id=base_agent['id'],
                meta_data={
                    "created_at": datetime.utcnow().isoformat(),
                    "client_info": {"client_id": client_id}
                }
            )

            self.db.add(conversation)
            self.db.commit()

            conv_info = {
                "id": conversation_id,
                "company_id": company_id,
                "customer_id": client_id,
                "current_agent_id": base_agent['id'],
                "meta_data": conversation.meta_data
            }

            self.conversation_cache[conversation_id] = conv_info
            return conv_info

        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            self.db.rollback()
            return None

    async def find_best_agent(
        self,
        company_id: str,
        query: str,
        current_agent_id: Optional[str] = None
    ) -> Tuple[Optional[str], float]:
        """Find best agent using pre-computed embeddings"""
        try:
            if company_id not in self.company_agents_cache:
                await self.initialize_company_agents(company_id)

            # Get query embedding from cache or compute
            query_embedding = await self.vector_store.get_query_embedding(query)
            
            # Search with embedding
            results = await self.vector_store.search(
                company_id=company_id,
                query_embedding=query_embedding,
                current_agent_id=current_agent_id
            )

            if not results:
                base_agent = await self.get_base_agent(company_id)
                return base_agent['id'] if base_agent else None, 0.0

            best_result = results[0]
            return best_result['agent_id'], best_result['score']

        except Exception as e:
            logger.error(f"Error finding best agent: {str(e)}")
            if current_agent_id:
                return current_agent_id, 0.0
            base_agent = await self.get_base_agent(company_id)
            return base_agent['id'] if base_agent else None, 0.0

    async def update_conversation(
        self,
        conversation_id: str,
        user_message: str,
        ai_response: str,
        agent_id: str,
        confidence_score: float = 0.0,
        tokens_used: Optional[int] = None,
        was_successful: bool = True,
        previous_agent_id: Optional[str] = None
    ) -> bool:
        try:
            response_time = time.time()  # Add response time tracking
            
            interaction = AgentInteraction(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                agent_id=agent_id,
                query=user_message,
                response=ai_response,
                confidence_score=confidence_score, 
                tokens_used=tokens_used,
                response_time=response_time,
                was_successful=was_successful,
                previous_agent_id=previous_agent_id,
                created_at=datetime.utcnow()
            )
            
            self.db.add(interaction)

            conversation = self.db.query(Conversation).filter_by(id=conversation_id).first()
            if conversation:
                conversation.current_agent_id = agent_id
                conversation.updated_at = datetime.utcnow()
                
                if conversation_id in self.conversation_cache:
                    self.conversation_cache[conversation_id].update({
                        "current_agent_id": agent_id,
                        "updated_at": conversation.updated_at.isoformat()
                    })

            self.db.commit()
            return True

        except Exception as e:
            logger.error(f"Error updating conversation: {str(e)}")
            self.db.rollback()
            return False
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        limit: int = 5
    ) -> List[Dict[str, str]]:
        """Get recent conversation history"""
        try:
            interactions = self.db.query(AgentInteraction).filter_by(
                conversation_id=conversation_id
            ).order_by(
                AgentInteraction.created_at.desc()
            ).limit(limit).all()

            context = []
            for interaction in reversed(interactions):
                context.extend([
                    {"role": "user", "content": interaction.query},
                    {"role": "assistant", "content": interaction.response}
                ])

            return context

        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
            return []

    async def cleanup_inactive_agents(self, days: int = 30) -> None:
        """Cleanup inactive agents and their data"""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            inactive_agents = self.db.query(Agent).filter(
                Agent.updated_at < cutoff,
                Agent.type != 'base'
            ).all()

            for agent in inactive_agents:
                # Delete from vector store
                await self.vector_store.delete_agent_data(agent.company_id, agent.id)
                
                # Clear caches
                self.agent_cache.pop(agent.id, None)
                if agent.company_id in self.company_agents_cache:
                    self.company_agents_cache[agent.company_id] = [
                        aid for aid in self.company_agents_cache[agent.company_id]
                        if aid != agent.id
                    ]

                # Delete from DB
                self.db.delete(agent)

            self.db.commit()
            logger.info(f"Cleaned up {len(inactive_agents)} inactive agents")

        except Exception as e:
            logger.error(f"Error cleaning up agents: {str(e)}")
            self.db.rollback()