# src/application/services/connection_service.py
import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from fastapi import WebSocket

from core.interfaces.services import IConnectionService, IAnalyticsService, IAgentService
from core.interfaces.repositories import IClientRepository, IAgentRepository
from core.entities.client import ClientSession, ConnectionState
from core.entities.analytics import SessionEvent, EventType
from database.config import get_db

logger = logging.getLogger(__name__)

class ConnectionService(IConnectionService):
    """Service for managing client connections"""
    
    def __init__(
        self,
        client_repository: IClientRepository,
        agent_repository: IAgentRepository,
        analytics_service: IAnalyticsService,
        agent_service: IAgentService,
        max_connections: int = 1000,
        max_requests_per_minute: int = 60
    ):
        self.client_repository = client_repository
        self.agent_repository = agent_repository
        self.analytics_service = analytics_service
        self.agent_service = agent_service
        
        # Configuration
        self.max_connections = max_connections
        self.max_requests_per_minute = max_requests_per_minute
        
        # Live state
        self.clients: Dict[str, ClientSession] = {}
        
        # Start maintenance task
        self._maintenance_task = asyncio.create_task(self._periodic_maintenance())
    
    async def connect_client(self, client_id: str, websocket: WebSocket) -> bool:
        """Accept and establish client connection"""
        try:
            # Check capacity
            if len(self.clients) >= self.max_connections:
                await websocket.close(code=1013, reason="Server at capacity")
                logger.warning(f"Connection rejected: {client_id} (capacity exceeded)")
                return False
            
            # Accept WebSocket connection
            await websocket.accept()
            
            # Create client session
            session = ClientSession(
                client_id=client_id,
                websocket=websocket,
                max_requests_per_minute=self.max_requests_per_minute
            )
            session.state = ConnectionState.CONNECTED
            
            # Store in active sessions
            self.clients[client_id] = session
            
            logger.info(f"Client connected: {client_id} ({len(self.clients)}/{self.max_connections})")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting client {client_id}: {e}")
            return False
    
    async def disconnect_client(self, client_id: str) -> None:
        """Disconnect client and cleanup resources"""
        session = self.clients.get(client_id)
        if not session:
            return
        
        try:
            # Update state
            session.state = ConnectionState.DISCONNECTING
            
            # Record analytics
            await self.analytics_service.record_disconnection(session)
            
            # Cleanup agent resources
            if session.agent_id:
                await self.agent_service.cleanup_agent_resources(client_id)
            
            # Close WebSocket if still open
            if not self._is_websocket_closed(session.websocket):
                await session.websocket.close()
            
            # Mark as disconnected
            session.state = ConnectionState.DISCONNECTED
            
        except Exception as e:
            logger.error(f"Error during disconnect {client_id}: {e}")
        finally:
            # Remove from active sessions
            self.clients.pop(client_id, None)
            logger.info(f"Client disconnected: {client_id} ({len(self.clients)}/{self.max_connections})")
    
    async def authenticate_client(self, client_id: str, api_key: str) -> bool:
        """Authenticate client with company API key"""
        session = self.clients.get(client_id)
        if not session:
            logger.warning(f"Authentication attempted for unknown client: {client_id}")
            return False
        
        try:
            # Authenticate via repository
            company = await self.client_repository.authenticate_company(api_key)
            if not company:
                await self._send_error(session, "Invalid API key")
                return False
            
            # Set company context
            session.set_company(company)
            
            # Record connection analytics
            await self.analytics_service.record_connection(session)
            
            # Send acknowledgment
            await self._send_json(session, {
                "type": "connection_ack",
                "company_name": company["name"],
                "client_id": client_id
            })
            
            logger.info(f"Client authenticated: {client_id} for {company['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Authentication error for {client_id}: {e}")
            await self._send_error(session, "Authentication failed")
            return False
    
    async def initialize_agent(self, client_id: str, agent_id: Optional[str] = None) -> bool:
        """Initialize AI agent resources for client"""
        session = self.clients.get(client_id)
        if not session or not session.company:
            return False
        
        try:
            # Initialize agent resources
            success = await self.agent_service.initialize_agent_resources(
                client_id, session.company["id"], agent_id
            )
            
            if success:
                session.set_agent_resources(agent_id, {"initialized": True})
                
                await self._send_json(session, {
                    "type": "agent_initialized",
                    "agent_id": agent_id,
                    "status": "ready"
                })
                
                logger.info(f"Agent initialized for {client_id}: {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error initializing agent for {client_id}: {e}")
            return False
    
    def get_client_session(self, client_id: str) -> Optional[ClientSession]:
        """Get client session by ID"""
        return self.clients.get(client_id)
    
    def get_active_clients(self) -> List[str]:
        """Get list of active client IDs"""
        return list(self.clients.keys())
    
    async def force_disconnect_client(self, client_id: str, reason: str = "Admin disconnect") -> bool:
        """Forcefully disconnect a client"""
        session = self.clients.get(client_id)
        if not session:
            return False
        
        try:
            # Send disconnect notice
            await self._send_json(session, {
                "type": "force_disconnect",
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Disconnect
            await self.disconnect_client(client_id)
            
            logger.warning(f"Force disconnected client: {client_id}, reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error force disconnecting {client_id}: {e}")
            return False
    
    # Helper methods
    async def _send_json(self, session: ClientSession, data: Dict) -> bool:
        """Send JSON data to client"""
        try:
            if self._is_websocket_closed(session.websocket):
                return False
            
            import json
            json_str = json.dumps(data)
            await asyncio.wait_for(session.websocket.send_text(json_str), timeout=5.0)
            return True
            
        except Exception as e:
            logger.error(f"Error sending JSON to {session.client_id}: {e}")
            return False
    
    async def _send_error(self, session: ClientSession, message: str) -> None:
        """Send error message to client"""
        await self._send_json(session, {
            "type": "error",
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def _is_websocket_closed(self, websocket: WebSocket) -> bool:
        """Check if WebSocket is closed"""
        try:
            # Check if custom flag is set
            if not getattr(websocket, '_peer_connected', True):
                return True
            
            # Check client state
            client_state = getattr(websocket, 'client_state', None)
            if client_state and hasattr(client_state, 'name'):
                return client_state.name != 'CONNECTED'
            
            return getattr(websocket, '_closed', False)
        except Exception:
            return False
    
    async def _periodic_maintenance(self) -> None:
        """Background maintenance task"""
        logger.info("Starting connection maintenance task")
        
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_stale_connections()
                
            except asyncio.CancelledError:
                logger.info("Maintenance task cancelled")
                break
            except Exception as e:
                logger.error(f"Maintenance task error: {e}")
    
    async def _cleanup_stale_connections(self) -> None:
        """Remove stale or broken connections"""
        current_time = datetime.utcnow()
        stale_clients = []
        
        for client_id, session in self.clients.items():
            # Check if WebSocket is closed
            if self._is_websocket_closed(session.websocket):
                stale_clients.append(client_id)
                continue
            
            # Check for stale sessions
            if session.is_stale(stale_threshold_minutes=30):
                stale_clients.append(client_id)
                continue
        
        # Cleanup stale clients
        for client_id in stale_clients:
            await self.disconnect_client(client_id)
        
        if stale_clients:
            logger.info(f"Cleaned up {len(stale_clients)} stale connections")