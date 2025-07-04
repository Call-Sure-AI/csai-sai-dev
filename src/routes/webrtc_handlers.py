# src/routes/webrtc_handlers.py (refactored with DI) - COMPLETE VERSION
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from dependency_injector.wiring import Provide, inject
from typing import Dict, Any, Optional
import logging
import json
import asyncio
import base64
from datetime import datetime

from di.container import Container
from core.interfaces.services import (
    IConnectionService, IConversationService, 
    IVoiceService, IAnalyticsService
)
# Fixed import path to match your structure
from core.application.dto.requests import MessageRequest
from core.application.dto.responses import ConnectionResponse, StatsResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/signal/{peer_id}/{company_api_key}")
@inject
async def websocket_endpoint(
    websocket: WebSocket,
    peer_id: str,
    company_api_key: str,
    connection_service: IConnectionService = Depends(Provide[Container.connection_service]),
    conversation_service: IConversationService = Depends(Provide[Container.conversation_service]),
    voice_service: IVoiceService = Depends(Provide[Container.voice_service])
):
    """Main WebRTC signaling endpoint with clean separation of concerns"""
    
    # Step 1: Connect client
    connected = await connection_service.connect_client(peer_id, websocket)
    if not connected:
        logger.warning(f"Failed to connect client: {peer_id}")
        return
    
    # Step 2: Authenticate
    authenticated = await connection_service.authenticate_client(peer_id, company_api_key)
    if not authenticated:
        await connection_service.disconnect_client(peer_id)
        logger.warning(f"Authentication failed for client: {peer_id}")
        return
    
    # Step 3: Initialize agent
    agent_initialized = await connection_service.initialize_agent(peer_id)
    if not agent_initialized:
        await connection_service.disconnect_client(peer_id)
        logger.warning(f"Agent initialization failed for client: {peer_id}")
        return
    
    try:
        # Main message processing loop
        while True:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0  # 30 second timeout
                )
                
                message_type = data.get("type")
                
                if message_type == "message":
                    # Process text message
                    await _handle_text_message(
                        peer_id, data, conversation_service, websocket
                    )
                
                elif message_type == "audio_chunk":
                    # Process audio data
                    await _handle_audio_message(
                        peer_id, data, voice_service
                    )
                
                elif message_type == "start_voice":
                    # Start voice call
                    await _handle_voice_start(
                        peer_id, data, voice_service, websocket
                    )
                
                elif message_type == "end_voice":
                    # End voice call
                    await _handle_voice_end(
                        peer_id, voice_service, websocket
                    )
                
                elif message_type == "ping":
                    # Heartbeat
                    await websocket.send_json({"type": "pong"})
                
                else:
                    logger.warning(f"Unknown message type: {message_type}")
                    
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})
                
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from {peer_id}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {peer_id}")
    except Exception as e:
        logger.error(f"Unexpected error for client {peer_id}: {e}")
    finally:
        # Cleanup
        await connection_service.disconnect_client(peer_id)

async def _handle_text_message(
    peer_id: str, 
    data: Dict[str, Any], 
    conversation_service: IConversationService,
    websocket: WebSocket
) -> None:
    """Handle text message processing"""
    try:
        message = data.get("message", "")
        metadata = data.get("metadata")
        
        if not message.strip():
            return
        
        # Process message and stream response
        chunk_number = 0
        async for token in conversation_service.process_message(peer_id, message, metadata):
            chunk_number += 1
            await websocket.send_json({
                "type": "stream_chunk",
                "content": token,
                "chunk_number": chunk_number
            })
        
        # Send end marker
        await websocket.send_json({
            "type": "stream_end",
            "total_chunks": chunk_number
        })
        
    except Exception as e:
        logger.error(f"Error processing text message for {peer_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "message": "Error processing message"
        })

async def _handle_audio_message(
    peer_id: str,
    data: Dict[str, Any],
    voice_service: IVoiceService
) -> None:
    """Handle audio chunk processing"""
    try:
        # Extract audio data (would be base64 encoded)
        audio_data = data.get("audio_data", "")
        if audio_data:
            # Decode if needed and process
            decoded_audio = base64.b64decode(audio_data)
            await voice_service.process_audio_chunk(peer_id, decoded_audio)
            
    except Exception as e:
        logger.error(f"Error processing audio for {peer_id}: {e}")

async def _handle_voice_start(
    peer_id: str,
    data: Dict[str, Any],
    voice_service: IVoiceService,
    websocket: WebSocket
) -> None:
    """Handle voice call start"""
    try:
        # Define callback for voice responses
        async def voice_callback(client_id: str, text: str):
            # Convert text to speech and send audio
            audio_data = await voice_service.synthesize_speech(client_id, text)
            if audio_data:
                encoded_audio = base64.b64encode(audio_data).decode()
                await websocket.send_json({
                    "type": "audio_response",
                    "audio_data": encoded_audio
                })
        
        # Start voice call
        success = await voice_service.start_voice_call(peer_id, voice_callback)
        
        await websocket.send_json({
            "type": "voice_started" if success else "voice_start_failed"
        })
        
    except Exception as e:
        logger.error(f"Error starting voice call for {peer_id}: {e}")
        await websocket.send_json({
            "type": "voice_start_failed",
            "error": str(e)
        })

async def _handle_voice_end(
    peer_id: str,
    voice_service: IVoiceService,
    websocket: WebSocket
) -> None:
    """Handle voice call end"""
    try:
        duration = await voice_service.end_voice_call(peer_id)
        
        await websocket.send_json({
            "type": "voice_ended",
            "duration": duration
        })
        
    except Exception as e:
        logger.error(f"Error ending voice call for {peer_id}: {e}")

@router.get("/stats")
@inject
async def get_webrtc_stats(
    analytics_service: IAnalyticsService = Depends(Provide[Container.analytics_service])
) -> Dict[str, Any]:
    """Get WebRTC system statistics"""
    try:
        stats = await analytics_service.get_live_stats()
        return {
            "timestamp": stats.timestamp.isoformat(),
            "total_connections": stats.total_connections,
            "voice_calls_active": stats.voice_calls_active,
            "companies_active": stats.companies_active,
            "processing_utilization": stats.calculate_processing_utilization()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")

@router.get("/peers/{company_api_key}")
@inject
async def get_active_peers(
    company_api_key: str,
    connection_service: IConnectionService = Depends(Provide[Container.connection_service])
) -> Dict[str, Any]:
    """Get active peers for a company"""
    try:
        active_clients = connection_service.get_active_clients()
        
        # Filter by company if needed (requires additional logic in service)
        return {
            "company_api_key": company_api_key,
            "active_peers": active_clients,
            "total_count": len(active_clients)
        }
    except Exception as e:
        logger.error(f"Error getting active peers: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving peers")

@router.get("/health")
async def webrtc_health_check():
    """WebRTC service health check"""
    return {
        "status": "healthy", 
        "service": "webrtc",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.websocket("/health")
async def websocket_health_check(websocket: WebSocket):
    """WebSocket health check endpoint"""
    await websocket.accept()
    await websocket.send_json({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "webrtc"
    })
    await websocket.close()

# src/routes/admin_routes_handlers.py (refactored with DI)
from fastapi import APIRouter, Depends, HTTPException, Query
from dependency_injector.wiring import Provide, inject
from typing import List, Optional, Dict, Any
import logging

from di.container import Container
from core.interfaces.services import IConnectionService, IAnalyticsService
from core.application.dto.responses import ClientSummaryResponse, HealthResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/clients")
@inject
async def get_all_clients(
    connection_service: IConnectionService = Depends(Provide[Container.connection_service])
) -> List[Dict[str, Any]]:
    """Get all connected clients summary"""
    try:
        client_ids = connection_service.get_active_clients()
        clients_summary = []
        
        for client_id in client_ids:
            session = connection_service.get_client_session(client_id)
            if session:
                # Handle different session structures gracefully
                voice_active = False
                if hasattr(session, 'voice_state'):
                    voice_active = session.voice_state.value != "inactive"
                elif hasattr(session, 'is_voice_call'):
                    voice_active = session.is_voice_call
                
                clients_summary.append({
                    "client_id": session.client_id,
                    "company_name": session.company["name"] if session.company else "Unknown",
                    "connection_time": session.connection_time.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "message_count": session.message_count,
                    "is_voice_call": voice_active,
                    "session_duration": session.get_session_duration(),
                    "total_tokens": getattr(session, 'total_tokens', 0)
                })
        
        return clients_summary
        
    except Exception as e:
        logger.error(f"Error getting clients: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving client list")

@router.delete("/clients/{client_id}")
@inject
async def force_disconnect_client(
    client_id: str,
    reason: str = Query("Admin disconnect"),
    connection_service: IConnectionService = Depends(Provide[Container.connection_service])
) -> Dict[str, Any]:
    """Forcefully disconnect a client"""
    try:
        success = await connection_service.force_disconnect_client(client_id, reason)
        
        if not success:
            raise HTTPException(status_code=404, detail="Client not found")
        
        return {
            "message": f"Client {client_id} disconnected", 
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error force disconnecting client {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Error disconnecting client")

@router.get("/health")
@inject
async def system_health_check(
    analytics_service: IAnalyticsService = Depends(Provide[Container.analytics_service]),
    connection_service: IConnectionService = Depends(Provide[Container.connection_service])
) -> Dict[str, Any]:
    """Comprehensive system health check"""
    try:
        # Get live stats
        stats = await analytics_service.get_live_stats()
        active_clients = connection_service.get_active_clients()
        
        # Determine health status
        status = "healthy"
        warnings = []
        
        # Check utilization (assuming max 1000 connections)
        connection_utilization = len(active_clients) / 1000 * 100
        if connection_utilization > 90:
            status = "warning"
            warnings.append("High connection utilization")
        
        processing_utilization = stats.calculate_processing_utilization()
        if processing_utilization > 85:
            status = "warning"
            warnings.append("High processing load")
        
        return {
            "status": status,
            "timestamp": stats.timestamp.isoformat(),
            "total_connections": len(active_clients),
            "processing_active": stats.processing_active,
            "processing_utilization": processing_utilization,
            "connection_utilization": connection_utilization,
            "warnings": warnings,
            "companies_active": stats.companies_active,
            "voice_calls_active": stats.voice_calls_active
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "total_connections": 0,
            "processing_active": 0,
            "warnings": [f"Health check failed: {str(e)}"]
        }

@router.get("/analytics/company/{company_id}")
@inject
async def get_company_analytics(
    company_id: str,
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    analytics_service: IAnalyticsService = Depends(Provide[Container.analytics_service])
) -> Dict[str, Any]:
    """Get analytics report for company"""
    try:
        report = await analytics_service.get_company_usage_report(
            company_id, start_date, end_date
        )
        return report
        
    except Exception as e:
        logger.error(f"Error getting company analytics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving analytics")

@router.get("/stats")
@inject
async def get_system_stats(
    analytics_service: IAnalyticsService = Depends(Provide[Container.analytics_service])
) -> Dict[str, Any]:
    """Get current system statistics"""
    try:
        stats = await analytics_service.get_live_stats()
        return {
            "timestamp": stats.timestamp.isoformat(),
            "connections": {
                "total": stats.total_connections,
                "initialized": getattr(stats, 'initialized_connections', 0),
                "voice_calls_active": stats.voice_calls_active
            },
            "companies_active": stats.companies_active,
            "processing": {
                "active": stats.processing_active,
                "capacity": getattr(stats, 'processing_capacity', 20),
                "utilization": stats.calculate_processing_utilization()
            },
            "memory_usage_mb": getattr(stats, 'memory_usage_mb', 0.0)
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving system statistics")

@router.get("/clients/{client_id}")
@inject
async def get_client_details(
    client_id: str,
    connection_service: IConnectionService = Depends(Provide[Container.connection_service])
) -> Dict[str, Any]:
    """Get detailed information about a specific client"""
    try:
        session = connection_service.get_client_session(client_id)
        if not session:
            raise HTTPException(status_code=404, detail="Client not found")
        
        return {
            "client_id": session.client_id,
            "company": session.company,
            "connection_time": session.connection_time.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "session_duration": session.get_session_duration(),
            "message_count": session.message_count,
            "total_tokens": getattr(session, 'total_tokens', 0),
            "is_voice_call": getattr(session, 'is_voice_call', False),
            "voice_duration": getattr(session, 'get_voice_duration', lambda: 0)(),
            "initialized": getattr(session, 'initialized', False)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting client details for {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving client details")

@router.post("/broadcast")
@inject
async def broadcast_message(
    message: Dict[str, Any],
    company_id: Optional[str] = Query(None, description="Optional company filter"),
    exclude_client: Optional[str] = Query(None, description="Client to exclude"),
    connection_service: IConnectionService = Depends(Provide[Container.connection_service])
) -> Dict[str, Any]:
    """Broadcast message to connected clients"""
    try:
        # This would require implementing broadcast functionality in connection service
        sent_count = await connection_service.broadcast_message(
            message, company_filter=company_id, exclude_client=exclude_client
        )
        
        return {
            "message": "Broadcast sent",
            "recipients": sent_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        raise HTTPException(status_code=500, detail="Error broadcasting message")