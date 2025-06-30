# src/routes/__init__.py

"""
Routes module for AI Backend application.

This module contains all the route handlers organized by functionality:
- WebRTC signaling and real-time communication
- Admin endpoints for system management
- Health check endpoints
- Voice processing routes
- Calendar integration routes
- Data processing and RAG routes
- External service integrations (Twilio, Exotel)
- Vector store operations
- WebSocket handlers

All routes are designed to work with the dependency injection container
and follow clean architecture principles.
"""

from .healthcheck_handlers import healthcheck_router
from .webrtc_handlers import router as webrtc_router
from .admin_routes_handlers import router as admin_router
from .twilio_handlers import router as twilio_router
from .exotel_routes_handlers import router as exotel_router
from .voice_routes_handlers import router as voice_router
from .calendar_routes_handlers import router as calendar_router
from .data_routes_handlers import router as data_router
from .rag_routes_handlers import router as rag_router
from .vector_store_handlers import router as vector_store_router
from .websocket_handlers import router as websocket_router
from .google_sheets_routes import router as google_sheets_router

# Export all routers for easy importing
__all__ = [
    "healthcheck_router",
    "webrtc_router", 
    "admin_router",
    "twilio_router",
    "exotel_router",
    "voice_router",
    "calendar_router",
    "data_router",
    "rag_router",
    "vector_store_router",
    "websocket_router",
    "google_sheets_router"
]

# Route prefixes for consistent API structure
ROUTE_PREFIXES = {
    "health": "/health",
    "webrtc": "/webrtc", 
    "admin": "/admin",
    "twilio": "/twilio",
    "exotel": "/exotel",
    "voice": "/voice",
    "calendar": "/calendar",
    "data": "/data",
    "rag": "/rag",
    "vector_store": "/vector-store",
    "websocket": "/ws",
    "google_sheets": "/google-sheets"
}

# Route tags for OpenAPI documentation
ROUTE_TAGS = {
    "health": ["Health"],
    "webrtc": ["WebRTC"],
    "admin": ["Admin"],
    "twilio": ["Twilio"],
    "exotel": ["Exotel"], 
    "voice": ["Voice"],
    "calendar": ["Calendar"],
    "data": ["Data"],
    "rag": ["RAG"],
    "vector_store": ["Vector Store"],
    "websocket": ["WebSocket"],
    "google_sheets": ["Google Sheets"]
}