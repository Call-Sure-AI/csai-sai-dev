# managers/connection_manager.py

import logging
import asyncio
import time
import json
from typing import Dict, Optional, List, Any, Callable
from datetime import datetime, timedelta
from uuid import UUID
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship
from sqlalchemy.exc import SQLAlchemyError

from fastapi import WebSocket
from database.models import Company, Conversation, Agent, Document
from managers.agent_manager import AgentManager
from services.vector_store.qdrant_service import QdrantService
from services.rag.rag_service import RAGService
from services.ticket_service import AutoTicketService

logger = logging.getLogger(__name__)

# ====================
# DATABASE MODELS FOR ANALYTICS
# ====================

Base = declarative_base()

class CompanyUsageMetrics(Base):
    """Daily aggregated usage metrics per company for billing/BI."""
    __tablename__ = "company_usage_metrics"
    
    id = Column(String, primary_key=True)
    company_id = Column(String, ForeignKey("companies.id"), nullable=False)
    date = Column(DateTime, nullable=False)  # Daily aggregation
    
    # Connection metrics
    total_connections = Column(Integer, default=0)
    peak_concurrent_connections = Column(Integer, default=0)
    connection_hours = Column(Float, default=0.0)  # For billing
    
    # Message metrics
    total_messages = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    # Voice call metrics
    total_voice_calls = Column(Integer, default=0)
    total_voice_minutes = Column(Float, default=0.0)  # For billing
    
    # Business metrics
    tickets_created = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    
    # Performance metrics
    avg_response_time = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes for fast queries
    __table_args__ = (
        Index('idx_company_date', 'company_id', 'date'),
        Index('idx_date', 'date'),
    )

class SessionEvent(Base):
    """Individual session events for detailed analytics."""
    __tablename__ = "session_events"
    
    id = Column(String, primary_key=True)
    company_id = Column(String, ForeignKey("companies.id"), nullable=False)
    client_id = Column(String, nullable=False)
    event_type = Column(String, nullable=False)  # connect, disconnect, message, voice_start, voice_end, error
    
    # Event data
    timestamp = Column(DateTime, default=datetime.utcnow)
    session_duration = Column(Float, nullable=True)  # For disconnect events
    message_count = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    voice_duration = Column(Float, nullable=True)  # For voice_end events
    error_message = Column(Text, nullable=True)
    
    # Indexes for analytics queries
    __table_args__ = (
        Index('idx_company_timestamp', 'company_id', 'timestamp'),
        Index('idx_event_type_timestamp', 'event_type', 'timestamp'),
    )

# ====================
# JSON ENCODER
# ====================

class UUIDEncoder(json.JSONEncoder):
    """JSON encoder for UUID and datetime objects."""
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# ====================
# CLIENT SESSION (LIVE STATE ONLY)
# ====================

class ClientSession:
    """Live client session - single source of truth for current state only."""
    
    def __init__(self, client_id: str, websocket: WebSocket):
        # Core identification
        self.client_id = client_id
        self.websocket = websocket
        
        # Connection state (live only)
        self.connected = True
        self.initialized = False
        self.connection_time = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Session data (live only)
        self.company: Optional[Dict[str, Any]] = None
        self.conversation: Optional[Conversation] = None
        self.agent_id: Optional[str] = None
        self.agent_resources: Optional[Dict[str, Any]] = None
        
        # Current session metrics (live only)
        self.message_count = 0
        self.total_tokens = 0
        self.request_times: List[float] = []
        
        # Voice call state (live only)
        self.is_voice_call = False
        self.voice_start_time: Optional[datetime] = None
        self.voice_callback: Optional[Callable] = None
        
        # WebSocket state
        setattr(websocket, '_peer_connected', True)

    def update_activity(self, tokens: int = 0):
        """Update live activity metrics."""
        self.last_activity = datetime.utcnow()
        self.message_count += 1
        self.total_tokens += tokens

    def set_company(self, company_data: Dict[str, Any]):
        self.company = company_data

    def set_conversation(self, conversation: Conversation):
        self.conversation = conversation

    def set_agent_resources(self, agent_id: str, resources: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_resources = resources
        self.initialized = True

    def start_voice_call(self, voice_callback: Optional[Callable] = None):
        self.is_voice_call = True
        self.voice_start_time = datetime.utcnow()
        self.voice_callback = voice_callback

    def end_voice_call(self) -> float:
        if not self.is_voice_call or not self.voice_start_time:
            return 0.0
        
        duration = (datetime.utcnow() - self.voice_start_time).total_seconds()
        self.is_voice_call = False
        self.voice_start_time = None
        self.voice_callback = None
        return duration

    def check_rate_limit(self, max_requests: int, window_seconds: int) -> bool:
        now = time.time()
        cutoff = now - window_seconds
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        if len(self.request_times) >= max_requests:
            return False
        
        self.request_times.append(now)
        return True

    def is_websocket_closed(self) -> bool:
        try:
            if not getattr(self.websocket, '_peer_connected', True):
                return True
            client_state = getattr(self.websocket, 'client_state', None)
            if client_state and hasattr(client_state, 'name'):
                return client_state.name != 'CONNECTED'
            return getattr(self.websocket, '_closed', False)
        except Exception:
            return False

    def close_websocket(self):
        setattr(self.websocket, '_peer_connected', False)
        self.connected = False

    def get_session_duration(self) -> float:
        return (datetime.utcnow() - self.connection_time).total_seconds()

    def get_voice_duration(self) -> float:
        if self.voice_start_time:
            return (datetime.utcnow() - self.voice_start_time).total_seconds()
        return 0.0

# ====================
# ANALYTICS SERVICE
# ====================

class AnalyticsService:
    """Handles persistence of metrics to database for historical analytics."""
    
    def __init__(self, db: Session):
        self.db = db

    async def record_connection(self, client: ClientSession):
        """Record new connection event."""
        if not client.company:
            return
        
        try:
            event = SessionEvent(
                id=f"{client.client_id}_{int(time.time())}",
                company_id=client.company["id"],
                client_id=client.client_id,
                event_type="connect",
                timestamp=client.connection_time
            )
            self.db.add(event)
            await self._update_daily_metrics(client.company["id"], connections=1)
            self.db.commit()
        except Exception as e:
            logger.error(f"Error recording connection: {e}")
            self.db.rollback()

    async def record_disconnection(self, client: ClientSession):
        """Record disconnection event with session data."""
        if not client.company:
            return
        
        try:
            session_duration = client.get_session_duration()
            
            event = SessionEvent(
                id=f"{client.client_id}_disconnect_{int(time.time())}",
                company_id=client.company["id"],
                client_id=client.client_id,
                event_type="disconnect",
                session_duration=session_duration,
                message_count=client.message_count,
                token_count=client.total_tokens
            )
            self.db.add(event)
            
            # Update daily metrics
            await self._update_daily_metrics(
                client.company["id"],
                connection_hours=session_duration / 3600,  # Convert to hours
                messages=client.message_count,
                tokens=client.total_tokens
            )
            self.db.commit()
        except Exception as e:
            logger.error(f"Error recording disconnection: {e}")
            self.db.rollback()

    async def record_voice_call(self, client: ClientSession, duration: float):
        """Record voice call completion."""
        if not client.company:
            return
        
        try:
            event = SessionEvent(
                id=f"{client.client_id}_voice_{int(time.time())}",
                company_id=client.company["id"],
                client_id=client.client_id,
                event_type="voice_end",
                voice_duration=duration
            )
            self.db.add(event)
            
            await self._update_daily_metrics(
                client.company["id"],
                voice_calls=1,
                voice_minutes=duration / 60
            )
            self.db.commit()
        except Exception as e:
            logger.error(f"Error recording voice call: {e}")
            self.db.rollback()

    async def record_message(self, client: ClientSession, tokens: int, response_time: float):
        """Record message processing."""
        if not client.company:
            return
        
        try:
            await self._update_daily_metrics(
                client.company["id"],
                messages=1,
                tokens=tokens,
                avg_response_time=response_time
            )
            self.db.commit()
        except Exception as e:
            logger.error(f"Error recording message: {e}")
            self.db.rollback()

    async def record_ticket_creation(self, company_id: str):
        """Record ticket creation."""
        try:
            await self._update_daily_metrics(company_id, tickets=1)
            self.db.commit()
        except Exception as e:
            logger.error(f"Error recording ticket: {e}")
            self.db.rollback()

    async def record_error(self, company_id: str, error_message: str):
        """Record error occurrence."""
        try:
            event = SessionEvent(
                id=f"error_{company_id}_{int(time.time())}",
                company_id=company_id,
                client_id="system",
                event_type="error",
                error_message=error_message
            )
            self.db.add(event)
            
            await self._update_daily_metrics(company_id, errors=1)
            self.db.commit()
        except Exception as e:
            logger.error(f"Error recording error: {e}")
            self.db.rollback()

    async def _update_daily_metrics(self, company_id: str, **kwargs):
        """Update or create daily aggregated metrics."""
        today = datetime.utcnow().date()
        
        # Get or create today's metrics
        metrics = self.db.query(CompanyUsageMetrics).filter_by(
            company_id=company_id,
            date=today
        ).first()
        
        if not metrics:
            metrics = CompanyUsageMetrics(
                id=f"{company_id}_{today}",
                company_id=company_id,
                date=today
            )
            self.db.add(metrics)
        
        # Update metrics
        for key, value in kwargs.items():
            if key == "connections":
                metrics.total_connections += value
            elif key == "connection_hours":
                metrics.connection_hours += value
            elif key == "messages":
                metrics.total_messages += value
            elif key == "tokens":
                metrics.total_tokens += value
            elif key == "voice_calls":
                metrics.total_voice_calls += value
            elif key == "voice_minutes":
                metrics.total_voice_minutes += value
            elif key == "tickets":
                metrics.tickets_created += value
            elif key == "errors":
                metrics.errors_count += value
            elif key == "avg_response_time":
                # Update running average
                current_messages = metrics.total_messages or 1
                current_avg = metrics.avg_response_time or 0
                metrics.avg_response_time = (
                    (current_avg * (current_messages - 1) + value) / current_messages
                )

# ====================
# CONNECTION MANAGER (CLEAN LIVE STATE)
# ====================

class ConnectionManager:
    """
    Professional-grade connection manager with clean live state and database analytics.
    
    Architecture:
    - Live State: Fast, memory-efficient ClientSession for current operations
    - Historical State: Database persistence for billing, BI, and analytics
    """

    def __init__(self, vector_store: Optional[QdrantService] = None, 
                 max_connections: int = 1000, max_requests_per_minute: int = 60):
        # Core services
        self.vector_store = vector_store or QdrantService()
        
        # Live state only - single source of truth
        self.clients: Dict[str, ClientSession] = {}
        
        # Configuration
        self.max_connections = max_connections
        self.max_requests_per_minute = max_requests_per_minute
        self.rate_limit_window = 60
        
        # Performance controls
        self._processing_semaphore = asyncio.Semaphore(20)
        
        # System startup time
        self._start_time = datetime.utcnow()
        
        # Maintenance task
        self._cleanup_task = asyncio.create_task(self._periodic_maintenance())
        
        logger.info(f"ConnectionManager initialized: {max_connections} max connections, {max_requests_per_minute}/min rate limit")

    # ====================
    # CONNECTION LIFECYCLE
    # ====================

    async def connect(self, websocket: WebSocket, client_id: str, db: Session) -> bool:
        """Accept new connection with analytics recording."""
        if len(self.clients) >= self.max_connections:
            await websocket.close(code=1013, reason="Server at capacity")
            logger.warning(f"Connection rejected: {client_id} (capacity: {len(self.clients)}/{self.max_connections})")
            return False
        
        try:
            await websocket.accept()
            client = ClientSession(client_id, websocket)
            self.clients[client_id] = client
            
            logger.info(f"Client connected: {client_id} ({len(self.clients)}/{self.max_connections})")
            return True
            
        except Exception as e:
            logger.error(f"Error accepting connection {client_id}: {e}")
            return False

    async def disconnect(self, client_id: str, db: Session) -> None:
        """Handle disconnection with analytics recording."""
        client = self.clients.get(client_id)
        if not client:
            return
        
        try:
            # Record analytics before cleanup
            analytics = AnalyticsService(db)
            await analytics.record_disconnection(client)
            
            # Handle voice call cleanup
            if client.is_voice_call:
                voice_duration = client.end_voice_call()
                await analytics.record_voice_call(client, voice_duration)
            
            # Close WebSocket
            if not client.is_websocket_closed():
                client.close_websocket()
                await client.websocket.close()
                
        except Exception as e:
            logger.error(f"Error during disconnect {client_id}: {e}")
        
        # Clean up live state
        self.clients.pop(client_id, None)
        logger.info(f"Client disconnected: {client_id} ({len(self.clients)}/{self.max_connections})")

    async def authenticate_client(self, client_id: str, company_api_key: str, db: Session) -> bool:
        """Authenticate client and record connection analytics."""
        client = self.clients.get(client_id)
        if not client:
            return False
        
        try:
            # Find company
            company = db.query(Company).filter_by(api_key=company_api_key).first()
            if not company:
                await self._send_error(client, "Invalid API key")
                return False
            
            # Set company data
            client.set_company({
                "id": company.id,
                "name": company.name,
                "api_key": company_api_key
            })
            
            # Record connection analytics
            analytics = AnalyticsService(db)
            await analytics.record_connection(client)
            
            await self._send_json(client, {
                "type": "connection_ack",
                "company_name": company.name,
                "client_id": client_id
            })
            
            logger.info(f"Client authenticated: {client_id} for {company.name}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during authentication: {e}")
            db.rollback()
            await self._send_error(client, "Authentication failed")
            return False

    # ====================
    # AGENT INITIALIZATION  
    # ====================

    async def initialize_agent(self, client_id: str, agent_id: Optional[str], db: Session) -> bool:
        """Initialize agent resources."""
        client = self.clients.get(client_id)
        if not client or not client.company:
            return False
        
        try:
            # Initialize services
            agent_manager = AgentManager(db, self.vector_store)
            rag_service = RAGService(self.vector_store)
            
            # Get or create conversation
            conversation, _ = await agent_manager.get_or_create_conversation(
                client_id, client.company["id"], agent_id
            )
            client.set_conversation(conversation)
            
            # Get agent and build context
            agent_record = None
            if agent_id:
                agent_record = db.query(Agent).filter_by(
                    id=agent_id, 
                    company_id=client.company["id"]
                ).first()
            
            prompt = self._build_agent_prompt(agent_record)
            
            # Create RAG chain
            chain = await rag_service.create_qa_chain(
                company_id=client.company["id"],
                agent_id=agent_id,
                agent_prompt=prompt
            )
            
            # Set agent resources
            resources = {
                "agent_manager": agent_manager,
                "rag_service": rag_service,
                "chain": chain,
                "agent_record": agent_record,
                "prompt": prompt
            }
            client.set_agent_resources(agent_id, resources)
            
            await self._send_json(client, {
                "type": "agent_initialized",
                "agent_id": agent_id,
                "agent_name": agent_record.name if agent_record else "Default"
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing agent for {client_id}: {e}")
            db.rollback()
            return False

    def _build_agent_prompt(self, agent_record: Optional[Agent]) -> Optional[str]:
        """Build agent prompt from businessContext and roleDescription."""
        if not agent_record:
            return None
        
        additional_context = agent_record.additional_context or {}
        business_context = additional_context.get('businessContext', '').strip()
        role_description = additional_context.get('roleDescription', '').strip()
        
        if business_context and role_description:
            return f"{business_context}\n\nRole: {role_description}"
        elif business_context:
            return business_context
        elif role_description:
            return f"Role: {role_description}"
        else:
            return agent_record.prompt

    # ====================
    # VOICE CALL SUPPORT
    # ====================

    def start_voice_call(self, client_id: str, voice_callback: Optional[Callable] = None) -> bool:
        """Start voice call session."""
        client = self.clients.get(client_id)
        if not client:
            return False
        
        client.start_voice_call(voice_callback)
        logger.info(f"Voice call started: {client_id}")
        return True

    async def end_voice_call(self, client_id: str, db: Session) -> float:
        """End voice call and record analytics."""
        client = self.clients.get(client_id)
        if not client or not client.is_voice_call:
            return 0.0
        
        duration = client.end_voice_call()
        
        # Record analytics
        if client.company:
            analytics = AnalyticsService(db)
            await analytics.record_voice_call(client, duration)
        
        logger.info(f"Voice call ended: {client_id}, duration: {duration:.1f}s")
        return duration

    # ====================
    # MESSAGE PROCESSING
    # ====================

    async def process_message(self, client_id: str, message_data: Dict[str, Any], db: Session) -> str:
        """Unified message processing with analytics recording."""
        client = self.clients.get(client_id)
        if not client or not client.initialized:
            return ""
        
        # Rate limiting
        if not client.check_rate_limit(self.max_requests_per_minute, self.rate_limit_window):
            await self._send_error(client, f"Rate limit exceeded: {self.max_requests_per_minute} requests/minute")
            return ""
        
        start_time = time.time()
        full_response = ""
        
        async with self._processing_semaphore:
            try:
                query = message_data.get("message", "").strip()
                if not query:
                    await self._send_error(client, "Empty message")
                    return ""
                
                # Get conversation context
                agent_manager = client.agent_resources["agent_manager"]
                context = await agent_manager.get_conversation_context(client.conversation.id)
                
                # Generate response
                chain = client.agent_resources["chain"]
                rag_service = client.agent_resources["rag_service"]
                
                # Stream response
                msg_id = f"{client_id}_{int(time.time() * 1000)}"
                chunk_number = 0
                accumulated_text = ""
                
                async for token in rag_service.get_answer_with_chain(
                    chain=chain,
                    question=query,
                    conversation_context=context
                ):
                    if client.is_websocket_closed():
                        break
                    
                    chunk_number += 1
                    full_response += token
                    
                    # Send to WebSocket clients
                    if not client.is_voice_call:
                        success = await self._send_json(client, {
                            "type": "stream_chunk",
                            "text_content": token,
                            "chunk_number": chunk_number,
                            "msg_id": msg_id
                        })
                        if not success:
                            break
                    
                    # Handle voice synthesis
                    if client.is_voice_call and client.voice_callback:
                        accumulated_text += token
                        if any(char in accumulated_text for char in ['.', '!', '?', '\n']):
                            try:
                                await client.voice_callback(client_id, accumulated_text.strip())
                                accumulated_text = ""
                            except Exception as e:
                                logger.error(f"Voice callback error: {e}")
                
                # Final voice text
                if client.is_voice_call and client.voice_callback and accumulated_text.strip():
                    try:
                        await client.voice_callback(client_id, accumulated_text.strip())
                    except Exception as e:
                        logger.error(f"Final voice callback error: {e}")
                
                # Stream end
                if not client.is_voice_call:
                    await self._send_json(client, {
                        "type": "stream_end",
                        "msg_id": msg_id,
                        "total_chunks": chunk_number
                    })
                
                # Update live state and record analytics
                response_time = time.time() - start_time
                token_count = len(full_response.split())
                
                client.update_activity(token_count)
                
                if client.company:
                    analytics = AnalyticsService(db)
                    await analytics.record_message(client, token_count, response_time)
                
                # Post-process
                asyncio.create_task(self._post_process_conversation(
                    client, query, full_response, db
                ))
                
                return full_response
                
            except Exception as e:
                # Record error analytics
                if client.company:
                    analytics = AnalyticsService(db)
                    await analytics.record_error(client.company["id"], str(e))
                
                logger.error(f"Error processing message {client_id}: {e}")
                await self._send_error(client, "Error processing message")
                return ""

    async def _post_process_conversation(self, client: ClientSession, query: str, response: str, db: Session):
        """Handle conversation updates and ticket creation."""
        try:
            # Update conversation
            agent_manager = client.agent_resources["agent_manager"]
            await agent_manager.update_conversation(
                client.conversation.id, query, response, client.agent_id
            )
            
            # Create tickets
            ticket_service = AutoTicketService(db)
            interaction = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ]
            
            ticket_data = await ticket_service.analyze_conversation_for_tickets(
                client.conversation.id, interaction
            )
            
            if ticket_data:
                ticket = await ticket_service.create_ticket(ticket_data)
                
                # Record analytics
                if client.company:
                    analytics = AnalyticsService(db)
                    await analytics.record_ticket_creation(client.company["id"])
                
                # Notify client
                await self._send_json(client, {
                    "type": "ticket_created",
                    "ticket_id": str(ticket.id),
                    "title": ticket.title
                })
                
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")

    # ====================
    # LIVE METRICS (CALCULATED FROM CLIENT STATE)
    # ====================

    def get_live_stats(self) -> Dict[str, Any]:
        """Get current live statistics (calculated from ClientSession data)."""
        current_time = datetime.utcnow()
        uptime = (current_time - self._start_time).total_seconds()
        
        # Calculate from live client data
        total_clients = len(self.clients)
        initialized_clients = sum(1 for c in self.clients.values() if c.initialized)
        voice_calls_active = sum(1 for c in self.clients.values() if c.is_voice_call)
        
        # Company distribution
        companies = set()
        total_live_messages = 0
        total_live_tokens = 0
        
        for client in self.clients.values():
            if client.company:
                companies.add(client.company["id"])
            total_live_messages += client.message_count
            total_live_tokens += client.total_tokens
        
        return {
            "timestamp": current_time.isoformat(),
            "uptime_seconds": uptime,
            "connections": {
                "total": total_clients,
                "initialized": initialized_clients,
                "voice_calls_active": voice_calls_active,
                "max_connections": self.max_connections,
                "utilization": total_clients / self.max_connections
            },
            "companies_active": len(companies),
            "current_session": {
                "messages": total_live_messages,
                "tokens": total_live_tokens,
                "avg_messages_per_client": total_live_messages / max(total_clients, 1)
            },
            "processing": {
                "active": 20 - self._processing_semaphore._value,
                "capacity": 20
            }
        }

    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get live client information."""
        client = self.clients.get(client_id)
        if not client:
            return None
        
        return {
            "client_id": client_id,
            "connected": client.connected,
            "initialized": client.initialized,
            "connection_time": client.connection_time.isoformat(),
            "last_activity": client.last_activity.isoformat(),
            "session_duration_seconds": client.get_session_duration(),
            "message_count": client.message_count,
            "total_tokens": client.total_tokens,
            "is_voice_call": client.is_voice_call,
            "voice_duration_seconds": client.get_voice_duration(),
            "rate_limit_requests": len(client.request_times),
            "company": client.company
        }

    # ====================
    # COMMUNICATION
    # ====================

    async def _send_json(self, client: ClientSession, data: Dict[str, Any]) -> bool:
        """Send JSON data with error handling."""
        try:
            if client.is_websocket_closed():
                return False
            
            json_str = json.dumps(data, cls=UUIDEncoder)
            await asyncio.wait_for(client.websocket.send_text(json_str), timeout=5.0)
            return True
            
        except Exception as e:
            logger.error(f"Error sending JSON to {client.client_id}: {e}")
            return False

    async def _send_error(self, client: ClientSession, message: str):
        """Send error message."""
        await self._send_json(client, {
            "type": "error",
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })

    # ====================
    # HEALTH CHECK
    # ====================

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        current_time = datetime.utcnow()
        
        # Basic health metrics
        total_clients = len(self.clients)
        healthy_clients = 0
        stale_clients = 0
        voice_calls = 0
        processing_load = 20 - self._processing_semaphore._value
        
        # Check client health
        stale_threshold = current_time - timedelta(minutes=30)
        
        for client in self.clients.values():
            if client.last_activity < stale_threshold:
                stale_clients += 1
            else:
                healthy_clients += 1
            
            if client.is_voice_call:
                voice_calls += 1
        
        # System health status
        health_status = "healthy"
        warnings = []
        
        # Check capacity
        utilization = total_clients / self.max_connections
        if utilization > 0.9:
            health_status = "warning"
            warnings.append(f"High capacity utilization: {utilization:.1%}")
        
        # Check processing load
        processing_utilization = processing_load / 20
        if processing_utilization > 0.8:
            health_status = "warning"
            warnings.append(f"High processing load: {processing_utilization:.1%}")
        
        # Check stale connections
        if stale_clients > total_clients * 0.2:
            health_status = "warning"
            warnings.append(f"Many stale connections: {stale_clients}/{total_clients}")
        
        # Overall system health
        if utilization > 0.95 or processing_utilization > 0.9:
            health_status = "critical"
        
        return {
            "status": health_status,
            "timestamp": current_time.isoformat(),
            "uptime_seconds": (current_time - self._start_time).total_seconds(),
            "warnings": warnings,
            "connections": {
                "total": total_clients,
                "healthy": healthy_clients,
                "stale": stale_clients,
                "voice_calls_active": voice_calls,
                "capacity_utilization": utilization
            },
            "processing": {
                "active_tasks": processing_load,
                "capacity": 20,
                "utilization": processing_utilization
            },
            "memory": {
                "clients_in_memory": len(self.clients),
                "semaphore_available": self._processing_semaphore._value
            },
            "rate_limiting": {
                "max_requests_per_minute": self.max_requests_per_minute,
                "window_seconds": self.rate_limit_window
            }
        }

    async def get_detailed_health(self, db: Session) -> Dict[str, Any]:
        """Get detailed health including database analytics."""
        basic_health = await self.health_check()
        
        try:
            # Get recent analytics from database
            today = datetime.utcnow().date()
            yesterday = today - timedelta(days=1)
            
            # Today's metrics
            today_metrics = db.query(CompanyUsageMetrics).filter_by(date=today).all()
            yesterday_metrics = db.query(CompanyUsageMetrics).filter_by(date=yesterday).all()
            
            # Aggregate metrics
            today_totals = {
                "connections": sum(m.total_connections for m in today_metrics),
                "messages": sum(m.total_messages for m in today_metrics),
                "tokens": sum(m.total_tokens for m in today_metrics),
                "voice_calls": sum(m.total_voice_calls for m in today_metrics),
                "voice_minutes": sum(m.total_voice_minutes for m in today_metrics),
                "tickets": sum(m.tickets_created for m in today_metrics),
                "errors": sum(m.errors_count for m in today_metrics),
                "companies": len(today_metrics)
            }
            
            yesterday_totals = {
                "connections": sum(m.total_connections for m in yesterday_metrics),
                "messages": sum(m.total_messages for m in yesterday_metrics),
                "tokens": sum(m.total_tokens for m in yesterday_metrics),
                "voice_calls": sum(m.total_voice_calls for m in yesterday_metrics),
                "voice_minutes": sum(m.total_voice_minutes for m in yesterday_metrics),
                "tickets": sum(m.tickets_created for m in yesterday_metrics),
                "errors": sum(m.errors_count for m in yesterday_metrics),
                "companies": len(yesterday_metrics)
            }
            
            # Calculate trends
            trends = {}
            for key in today_totals:
                if yesterday_totals[key] > 0:
                    change = (today_totals[key] - yesterday_totals[key]) / yesterday_totals[key]
                    trends[f"{key}_change_percent"] = round(change * 100, 1)
                else:
                    trends[f"{key}_change_percent"] = 0
            
            # Add analytics to health data
            basic_health["analytics"] = {
                "today": today_totals,
                "yesterday": yesterday_totals,
                "trends": trends,
                "database_status": "connected"
            }
            
        except Exception as e:
            logger.error(f"Error getting database health: {e}")
            basic_health["analytics"] = {
                "database_status": "error",
                "error": str(e)
            }
        
        return basic_health

    # ====================
    # MAINTENANCE & CLEANUP
    # ====================

    async def _periodic_maintenance(self):
        """Background task for system maintenance."""
        logger.info("Starting periodic maintenance task")
        
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_stale_connections()
                await self._update_peak_metrics()
                
            except asyncio.CancelledError:
                logger.info("Maintenance task cancelled")
                break
            except Exception as e:
                logger.error(f"Maintenance task error: {e}")

    async def _cleanup_stale_connections(self):
        """Remove stale or broken connections."""
        current_time = datetime.utcnow()
        stale_threshold = current_time - timedelta(minutes=30)
        inactive_threshold = current_time - timedelta(hours=2)
        
        stale_clients = []
        
        for client_id, client in self.clients.items():
            # Check if WebSocket is closed
            if client.is_websocket_closed():
                stale_clients.append(client_id)
                continue
            
            # Check for very old inactive connections
            if client.last_activity < inactive_threshold:
                stale_clients.append(client_id)
                continue
            
            # Check for stale voice calls (over 1 hour)
            if client.is_voice_call and client.voice_start_time:
                voice_duration = (current_time - client.voice_start_time).total_seconds()
                if voice_duration > 3600:  # 1 hour
                    logger.warning(f"Cleaning up long voice call: {client_id} ({voice_duration:.0f}s)")
                    stale_clients.append(client_id)
        
        # Clean up stale clients
        for client_id in stale_clients:
            try:
                client = self.clients.get(client_id)
                if client:
                    if not client.is_websocket_closed():
                        client.close_websocket()
                        await client.websocket.close()
                    
                    self.clients.pop(client_id, None)
                    logger.info(f"Cleaned up stale client: {client_id}")
                    
            except Exception as e:
                logger.error(f"Error cleaning up client {client_id}: {e}")
        
        if stale_clients:
            logger.info(f"Cleaned up {len(stale_clients)} stale connections")

    async def _update_peak_metrics(self):
        """Update peak connection metrics for analytics."""
        # This would be called by analytics service if needed
        # For now, just log current peak
        current_connections = len(self.clients)
        
        # Track peak in memory for current session
        if not hasattr(self, '_session_peak_connections'):
            self._session_peak_connections = current_connections
        else:
            self._session_peak_connections = max(self._session_peak_connections, current_connections)

    async def force_disconnect_client(self, client_id: str, reason: str = "Admin disconnect") -> bool:
        """Forcefully disconnect a specific client."""
        client = self.clients.get(client_id)
        if not client:
            return False
        
        try:
            # Send disconnect notice
            await self._send_json(client, {
                "type": "force_disconnect",
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Close connection
            client.close_websocket()
            await client.websocket.close()
            self.clients.pop(client_id, None)
            
            logger.warning(f"Force disconnected client: {client_id}, reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error force disconnecting {client_id}: {e}")
            return False

    async def broadcast_message(self, message: Dict[str, Any], 
                               company_filter: Optional[str] = None,
                               exclude_client: Optional[str] = None) -> int:
        """Broadcast message to connected clients."""
        sent_count = 0
        
        for client_id, client in self.clients.items():
            # Skip excluded client
            if exclude_client and client_id == exclude_client:
                continue
            
            # Filter by company if specified
            if company_filter and (not client.company or client.company["id"] != company_filter):
                continue
            
            # Send message
            if await self._send_json(client, message):
                sent_count += 1
        
        logger.info(f"Broadcast message sent to {sent_count} clients")
        return sent_count

    # ====================
    # SHUTDOWN & CLEANUP
    # ====================

    async def shutdown(self, db: Session):
        """Graceful shutdown with cleanup."""
        logger.info("Starting ConnectionManager shutdown...")
        
        # Cancel maintenance task
        if hasattr(self, '_cleanup_task'):
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Notify all clients
        await self.broadcast_message({
            "type": "server_shutdown",
            "message": "Server is shutting down",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Disconnect all clients with analytics
        client_ids = list(self.clients.keys())
        for client_id in client_ids:
            await self.disconnect(client_id, db)
        
        logger.info(f"ConnectionManager shutdown complete. Disconnected {len(client_ids)} clients.")

    def get_session_peak_connections(self) -> int:
        """Get peak connections for current session."""
        return getattr(self, '_session_peak_connections', len(self.clients))

    # ====================
    # ADMIN ENDPOINTS
    # ====================

    def get_all_clients_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all connected clients for admin dashboard."""
        return [
            {
                "client_id": client_id,
                "company_name": client.company["name"] if client.company else "Unknown",
                "connection_time": client.connection_time.isoformat(),
                "last_activity": client.last_activity.isoformat(),
                "message_count": client.message_count,
                "is_voice_call": client.is_voice_call,
                "initialized": client.initialized,
                "session_duration": client.get_session_duration()
            }
            for client_id, client in self.clients.items()
        ]

    def get_company_clients(self, company_id: str) -> List[Dict[str, Any]]:
        """Get all clients for a specific company."""
        return [
            {
                "client_id": client_id,
                "connection_time": client.connection_time.isoformat(),
                "last_activity": client.last_activity.isoformat(),
                "message_count": client.message_count,
                "total_tokens": client.total_tokens,
                "is_voice_call": client.is_voice_call,
                "voice_duration": client.get_voice_duration(),
                "session_duration": client.get_session_duration()
            }
            for client_id, client in self.clients.items()
            if client.company and client.company["id"] == company_id
        ]

    def get_rate_limit_status(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get rate limiting status for a client."""
        client = self.clients.get(client_id)
        if not client:
            return None
        
        now = time.time()
        cutoff = now - self.rate_limit_window
        recent_requests = [t for t in client.request_times if t > cutoff]
        
        return {
            "client_id": client_id,
            "requests_in_window": len(recent_requests),
            "max_requests": self.max_requests_per_minute,
            "window_seconds": self.rate_limit_window,
            "requests_remaining": max(0, self.max_requests_per_minute - len(recent_requests)),
            "reset_time": cutoff + self.rate_limit_window if recent_requests else now
        }

# ====================
# ANALYTICS QUERIES (DATABASE LAYER)
# ====================

class AnalyticsQueries:
    """Database queries for business intelligence and reporting."""
    
    @staticmethod
    def get_company_usage_report(db: Session, company_id: str, 
                                start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get comprehensive usage report for a company."""
        metrics = db.query(CompanyUsageMetrics).filter(
            CompanyUsageMetrics.company_id == company_id,
            CompanyUsageMetrics.date >= start_date.date(),
            CompanyUsageMetrics.date <= end_date.date()
        ).all()
        
        if not metrics:
            return {"error": "No data found for the specified period"}
        
        # Aggregate totals
        totals = {
            "total_connections": sum(m.total_connections for m in metrics),
            "total_messages": sum(m.total_messages for m in metrics),
            "total_tokens": sum(m.total_tokens for m in metrics),
            "total_voice_calls": sum(m.total_voice_calls for m in metrics),
            "total_voice_minutes": sum(m.total_voice_minutes for m in metrics),
            "total_connection_hours": sum(m.connection_hours for m in metrics),
            "tickets_created": sum(m.tickets_created for m in metrics),
            "errors_count": sum(m.errors_count for m in metrics)
        }
        
        # Daily breakdown
        daily_data = [
            {
                "date": m.date.isoformat(),
                "connections": m.total_connections,
                "messages": m.total_messages,
                "tokens": m.total_tokens,
                "voice_calls": m.total_voice_calls,
                "voice_minutes": m.total_voice_minutes,
                "connection_hours": m.connection_hours,
                "avg_response_time": m.avg_response_time
            }
            for m in sorted(metrics, key=lambda x: x.date)
        ]
        
        return {
            "company_id": company_id,
            "period": {
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "days": len(metrics)
            },
            "totals": totals,
            "daily_breakdown": daily_data
        }
    
    @staticmethod
    def get_system_overview(db: Session, date: datetime) -> Dict[str, Any]:
        """Get system-wide overview for a specific date."""
        metrics = db.query(CompanyUsageMetrics).filter_by(date=date.date()).all()
        
        if not metrics:
            return {"error": "No data found for the specified date"}
        
        return {
            "date": date.date().isoformat(),
            "companies_active": len(metrics),
            "total_connections": sum(m.total_connections for m in metrics),
            "total_messages": sum(m.total_messages for m in metrics),
            "total_tokens": sum(m.total_tokens for m in metrics),
            "total_voice_calls": sum(m.total_voice_calls for m in metrics),
            "total_voice_minutes": sum(m.total_voice_minutes for m in metrics),
            "total_tickets_created": sum(m.tickets_created for m in metrics),
            "total_errors": sum(m.errors_count for m in metrics),
            "avg_response_time": sum(m.avg_response_time for m in metrics) / len(metrics) if metrics else 0
        }