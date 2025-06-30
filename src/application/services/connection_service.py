# src/application/services/connection_service.py
"""
Connection service for managing client connections and authentication.
"""

import asyncio
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect

from core.interfaces.services import IConnectionService
from core.interfaces.repositories import IClientRepository
from core.entities.client import ClientSession, ConnectionState
from core.exceptions import (
    MaxConnectionsExceededException,
    AuthenticationFailedException,
    ClientNotFoundException
)

logger = logging.getLogger(__name__)

class ConnectionService(IConnectionService):
    """Service for managing client connections and sessions."""
    
    def __init__(
        self,
        client_repository: IClientRepository,
        max_connections: int = 1000,
        connection_timeout: int = 300,
        heartbeat_interval: int = 30
    ):
        self.client_repository = client_repository
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.heartbeat_interval = heartbeat_interval
        
        # In-memory session storage
        self.active_sessions: Dict[str, ClientSession] = {}
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
    
    async def connect_client(self, client_id: str, websocket: WebSocket) -> bool:
        """Connect a new client and create session."""
        try:
            # Check connection limits
            if len(self.active_sessions) >= self.max_connections:
                await websocket.close(code=1013, reason="Server at capacity")
                raise MaxConnectionsExceededException(
                    current_connections=len(self.active_sessions),
                    max_connections=self.max_connections
                )
            
            # Accept WebSocket connection
            await websocket.accept()
            
            # Create client session
            session = ClientSession(
                client_id=client_id,
                connection_state=ConnectionState.CONNECTED
            )
            session.set_websocket_reference(websocket)
            
            # Store session
            self.active_sessions[client_id] = session
            
            # Record connection event
            await self.client_repository.record_connection_event(
                session.session_id,
                "connection",
                {"client_id": client_id, "timestamp": datetime.utcnow().isoformat()}
            )
            
            logger.info(f"Client connected: {client_id} ({len(self.active_sessions)}/{self.max_connections})")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting client {client_id}: {e}")
            return False
    
    async def disconnect_client(self, client_id: str) -> None:
        """Disconnect client and cleanup session."""
        session = self.active_sessions.get(client_id)
        if not session:
            return
        
        try:
            # Update session state
            session.transition_to(ConnectionState.DISCONNECTING)
            
            # End any active voice call
            if session.voice_call_state.name != "INACTIVE":
                session.end_voice_call()
            
            # Close WebSocket if still open
            websocket = session.get_websocket_reference()
            if websocket and not self._is_websocket_closed(websocket):
                await websocket.close()
            
            # Record disconnection event
            await self.client_repository.record_connection_event(
                session.session_id,
                "disconnection",
                {
                    "client_id": client_id,
                    "session_duration": session.get_session_duration(),
                    "message_count": session.metrics.message_count,
                    "total_tokens": session.metrics.total_tokens
                }
            )
            
            # Update final state
            session.transition_to(ConnectionState.DISCONNECTED)
            
        except Exception as e:
            logger.error(f"Error during disconnect for {client_id}: {e}")
        finally:
            # Remove from active sessions
            self.active_sessions.pop(client_id, None)
            logger.info(f"Client disconnected: {client_id} ({len(self.active_sessions)}/{self.max_connections})")
    
    async def authenticate_client(self, client_id: str, api_key: str) -> bool:
        """Authenticate client with company API key."""
        session = self.active_sessions.get(client_id)
        if not session:
            raise ClientNotFoundException(client_id)
        
        try:
            # Authenticate via repository
            company = await self.client_repository.authenticate_company(api_key)
            if not company:
                await self._send_error(session, "Invalid API key")
                raise AuthenticationFailedException("Invalid API key")
            
            # Update session with company data
            session.authenticate(
                company_id=company["id"],
                company_data=company,
                api_key_hash=api_key[:8] + "..."  # Store partial key for logging
            )
            
            # Send authentication confirmation
            await self._send_json(session, {
                "type": "auth_success",
                "company_name": company["name"],
                "session_id": session.session_id,
                "limits": {
                    "max_requests_per_minute": company.get("max_requests_per_minute", 60)
                }
            })
            
            logger.info(f"Client authenticated: {client_id} for company {company['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Authentication error for {client_id}: {e}")
            await self._send_error(session, "Authentication failed")
            return False
    
    async def get_client_session(self, client_id: str) -> Optional[ClientSession]:
        """Get client session by ID."""
        return self.active_sessions.get(client_id)
    
    async def get_active_clients(self) -> List[str]:
        """Get list of active client IDs."""
        return list(self.active_sessions.keys())
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        authenticated_count = sum(
            1 for session in self.active_sessions.values()
            if session.authenticated
        )
        
        voice_calls_active = sum(
            1 for session in self.active_sessions.values()
            if session.voice_call_state.name == "ACTIVE"
        )
        
        return {
            "total_connections": len(self.active_sessions),
            "authenticated_connections": authenticated_count,
            "voice_calls_active": voice_calls_active,
            "capacity_used": len(self.active_sessions) / self.max_connections,
            "max_connections": self.max_connections
        }
    
    async def broadcast_message(self, message: Dict[str, Any], filter_func=None) -> int:
        """Broadcast message to all or filtered clients."""
        sent_count = 0
        for session in self.active_sessions.values():
            if filter_func is None or filter_func(session):
                if await self._send_json(session, message):
                    sent_count += 1
        return sent_count
    
    async def force_disconnect_client(self, client_id: str, reason: str = "Admin disconnect") -> bool:
        """Force disconnect a client."""
        session = self.active_sessions.get(client_id)
        if not session:
            return False
        
        try:
            await self._send_json(session, {
                "type": "force_disconnect",
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            await self.disconnect_client(client_id)
            logger.warning(f"Force disconnected client: {client_id}, reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error force disconnecting {client_id}: {e}")
            return False
    
    # Helper methods
    async def _send_json(self, session: ClientSession, data: Dict[str, Any]) -> bool:
        """Send JSON data to client."""
        try:
            websocket = session.get_websocket_reference()
            if not websocket or self._is_websocket_closed(websocket):
                return False
            
            import json
            json_str = json.dumps(data)
            await asyncio.wait_for(websocket.send_text(json_str), timeout=5.0)
            return True
            
        except Exception as e:
            logger.error(f"Error sending JSON to {session.client_id}: {e}")
            return False
    
    async def _send_error(self, session: ClientSession, message: str) -> None:
        """Send error message to client."""
        await self._send_json(session, {
            "type": "error",
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def _is_websocket_closed(self, websocket: WebSocket) -> bool:
        """Check if WebSocket is closed."""
        try:
            # Check various WebSocket state indicators
            if hasattr(websocket, 'client_state'):
                client_state = websocket.client_state
                if hasattr(client_state, 'name'):
                    return client_state.name != 'CONNECTED'
            
            # Check if custom closed flag is set
            if hasattr(websocket, '_closed'):
                return websocket._closed
            
            return False
        except Exception:
            return True
    
    async def _periodic_cleanup(self) -> None:
        """Background task to cleanup stale connections."""
        logger.info("Starting connection cleanup task")
        
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_stale_sessions()
                
            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
    
    async def _cleanup_stale_sessions(self) -> None:
        """Remove stale or broken sessions."""
        stale_clients = []
        current_time = datetime.utcnow()
        
        for client_id, session in self.active_sessions.items():
            # Check if WebSocket is closed
            websocket = session.get_websocket_reference()
            if websocket and self._is_websocket_closed(websocket):
                stale_clients.append(client_id)
                continue
            
            # Check for idle timeout
            if session.is_idle():
                stale_clients.append(client_id)
                continue
        
        # Cleanup stale clients
        for client_id in stale_clients:
            await self.disconnect_client(client_id)
        
        if stale_clients:
            logger.info(f"Cleaned up {len(stale_clients)} stale connections")
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor client heartbeats."""
        logger.info("Starting heartbeat monitor")
        
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Send heartbeat to all connected clients
                heartbeat_message = {
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await self.broadcast_message(heartbeat_message)
                
            except asyncio.CancelledError:
                logger.info("Heartbeat monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        logger.info("Shutting down connection service")
        
        # Cancel background tasks
        self._cleanup_task.cancel()
        self._heartbeat_task.cancel()
        
        # Disconnect all clients
        client_ids = list(self.active_sessions.keys())
        for client_id in client_ids:
            await self.disconnect_client(client_id)
        
        logger.info("Connection service shutdown complete")
