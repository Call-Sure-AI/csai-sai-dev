from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from config.settings_manager import get_settings, validate_startup_configuration
from config.logging_config import setup_logging
from di.container import Container, configure_container
from routes import webrtc_handlers, admin_routes_handlers
from middleware.error_handler import add_error_handlers

# Validate configuration first
validate_startup_configuration()
settings = get_settings()

# Setup logging
setup_logging(
    log_level=settings.log_level,
    log_file=settings.log_file,
    app_name=settings.app_name
)

logger = logging.getLogger(__name__)

# Configure DI container
configure_container()
container = Container()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    description="Modular AI Backend Service with Clean Architecture"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handlers
add_error_handlers(app)

# Wire DI container
container.wire(modules=[
    "routes.webrtc_handlers",
    "routes.admin_routes_handlers"
])

# Include routers
app.include_router(
    webrtc_handlers.router, 
    prefix="/api/v1/webrtc", 
    tags=["WebRTC"]
)
app.include_router(
    admin_routes_handlers.router, 
    prefix="/api/v1/admin", 
    tags=["Admin"]
)

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down application")
    # Cleanup would be handled by DI container

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy"}