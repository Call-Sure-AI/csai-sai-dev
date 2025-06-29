# FILE: src/app.py
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import logging

from managers.connection_manager import ConnectionManager
from services.vector_store.qdrant_service import QdrantService
from services.webrtc.manager import WebRTCManager

logger = logging.getLogger(__name__)

from config.settings import ALLOWED_ORIGINS, DEBUG, APP_PREFIX
from routes.healthcheck_handlers import healthcheck_router
from routes.webrtc_handlers import router as webrtc_router
from routes.admin_routes_handlers import router as admin_router
from routes.twilio_handlers import router as twilio_router
from routes.exotel_routes_handlers import router as exotel_router
from routes.ticket_routes import router as ticket_router
# from routes.ai_routes_handlers import ai_router
# from routes.voice_routes_handlers import voice_router
# from routes.calendar_routes_handlers import calendar_router
# from routes.google_sheets_routes import router as google_sheets_router
# from routes.websocket_handlers import router as websocket_router

logger.info("allow_origins: %s", ALLOWED_ORIGINS)
app = FastAPI(
    title="AI Backend",
    version="1.0.0",
    debug=DEBUG,
    description="AI Backend Service"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Root route redirects to docs
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

# Include routers
app.include_router(healthcheck_router, prefix=f"{APP_PREFIX}/health", tags=["Health"])
app.include_router(webrtc_router, prefix=f"{APP_PREFIX}/webrtc", tags=["WebRTC"])
app.include_router(admin_router, prefix=f"{APP_PREFIX}/admin", tags=["Admin"])
app.include_router(twilio_router, prefix=f"{APP_PREFIX}/twilio", tags=["Twilio"])
app.include_router(exotel_router, prefix=f"{APP_PREFIX}/exotel", tags=["Exotel"])
app.include_router(ticket_router, prefix=f"{APP_PREFIX}/tickets", tags=["Tickets"])
# app.include_router(ai_router, prefix=f"{APP_PREFIX}/ai", tags=["AI"])
# app.include_router(voice_router, prefix=f"{APP_PREFIX}/voice", tags=["Voice"])
# app.include_router(calendar_router, prefix=f"{APP_PREFIX}/calendar", tags=["Calendar"])
# app.include_router(google_sheets_router, prefix=f"{APP_PREFIX}/google_sheets", tags=["Google Sheets"])
# app.include_router(websocket_router, prefix=f"{APP_PREFIX}/ws", tags=["websocket"])

@app.on_event("startup")
async def startup_event():
    try:
        # Initialize vector store
        vector_store = QdrantService()
        logger.info("Vector store initialized")

        # Initialize connection manager WITHOUT a shared DB session
        # The manager will handle its own session lifecycle.
        connection_manager = ConnectionManager(vector_store)
        app.state.connection_manager = connection_manager
        logger.info("Connection manager initialized")

        # Initialize WebRTC manager
        webrtc_manager = WebRTCManager()
        webrtc_manager.connection_manager = connection_manager
        app.state.webrtc_manager = webrtc_manager
        logger.info("WebRTC manager initialized")

        # Initialize additional state
        app.state.response_cache = {}
        app.state.stream_sids = {}
        app.state.call_mappings = {}
        app.state.client_call_mapping = {}
        app.state.transcripts = {}

        logger.info("Initialized shared application state")
    except Exception as e:
        logger.critical(f"Failed to initialize application: {str(e)}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    try:
        # Close any active connections
        if hasattr(app.state, "connection_manager"):
            await app.state.connection_manager.close_all_connections()
        
        # Close WebRTC connections
        if hasattr(app.state, "webrtc_manager"):
            await app.state.webrtc_manager.close_all_connections()
            
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during application shutdown: {str(e)}")