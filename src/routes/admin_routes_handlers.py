# src/routes/admin_routes_handlers.py
"""
Admin routes for system management and monitoring.

This module provides administrative endpoints for:
- Company management and authentication
- Agent creation and management with document/image support
- Client connection management
- System health monitoring
- Analytics and usage reports
- Document and image upload/processing
- Broadcasting messages
- Performance metrics
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body, UploadFile, File, Form
from dependency_injector.wiring import Provide, inject
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import logging
import json
import uuid
import os
from datetime import datetime
from urllib.parse import urlparse, unquote

from di.container import Container
from core.interfaces.services import IConnectionService, IAnalyticsService
from core.application.dto.responses import (
    ClientSummaryResponse, HealthResponse, AnalyticsResponse, ErrorResponse
)
from database.config import get_db
from database.models import Company, Agent, Document, DocumentType, AgentType
from services.vector_store.qdrant_service import QdrantService
from services.vector_store.document_embedding import DocumentEmbeddingService
from services.vector_store.image_embedding import ImageEmbeddingService

router = APIRouter()
logger = logging.getLogger(__name__)

# ========================================
# COMPANY MANAGEMENT
# ========================================

@router.post("/companies")
async def create_company(
    company_data: dict,
    db: Session = Depends(get_db)
):
    """Create a new company with auto-generated API key"""
    try:
        # Generate a unique API key if not provided
        if 'api_key' not in company_data or not company_data['api_key']:
            company_data['api_key'] = str(uuid.uuid4())
        logger.info(f"Generated API key: {company_data['api_key']}")
        
        company = Company(**company_data)
        db.add(company)
        db.commit()
        db.refresh(company)
        
        return {
            "status": "success",
            "company": {
                "id": company.id,
                "name": company.name,
                "email": company.email,
                "api_key": company.api_key,
                "created_at": company.created_at.isoformat()
            }
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating company: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating company: {str(e)}")

@router.post("/companies/login")
async def login_company(
    login_data: dict,
    db: Session = Depends(get_db)
):
    """Authenticate company with email and API key"""
    try:
        # Validate email and API key
        company = db.query(Company).filter(
            Company.email == login_data.get("email"),
            Company.api_key == login_data.get("api_key")
        ).first()
        
        if not company:
            raise HTTPException(status_code=401, detail="Invalid email or API key")
        
        # Return company details upon successful login
        return {
            "status": "success",
            "company": {
                "id": company.id,
                "name": company.name,
                "email": company.email,
                "api_key": company.api_key,
                "settings": company.settings,
                "created_at": company.created_at.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during login: {str(e)}")

@router.get("/companies")
@inject
async def get_active_companies(
    connection_service: IConnectionService = Depends(Provide[Container.connection_service]),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get list of companies with active connections and database info"""
    try:
        # Get active connections data
        active_clients = connection_service.get_active_clients()
        companies_with_connections = {}
        
        for client_id in active_clients:
            session = connection_service.get_client_session(client_id)
            if session and session.company:
                company_id = session.company["id"]
                company_name = session.company["name"]
                
                if company_id not in companies_with_connections:
                    companies_with_connections[company_id] = {
                        "id": company_id,
                        "name": company_name,
                        "active_connections": 0,
                        "voice_calls": 0,
                        "total_messages": 0,
                        "total_tokens": 0
                    }
                
                companies_with_connections[company_id]["active_connections"] += 1
                companies_with_connections[company_id]["total_messages"] += session.message_count
                companies_with_connections[company_id]["total_tokens"] += getattr(session, 'total_tokens', 0)
                
                if getattr(session, 'is_voice_call', False):
                    companies_with_connections[company_id]["voice_calls"] += 1
        
        # Get all companies from database
        all_companies = db.query(Company).all()
        companies_list = []
        
        for company in all_companies:
            company_data = {
                "id": company.id,
                "name": company.name,
                "email": company.email,
                "active": company.active,
                "created_at": company.created_at.isoformat(),
                "total_agents": len(company.agents),
                "total_documents": len(company.documents),
                "active_connections": 0,
                "voice_calls": 0,
                "total_messages": 0,
                "total_tokens": 0
            }
            
            # Add live connection data if available
            if company.id in companies_with_connections:
                company_data.update(companies_with_connections[company.id])
            
            companies_list.append(company_data)
        
        return {
            "companies": companies_list,
            "total_companies": len(companies_list),
            "active_companies": len(companies_with_connections),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting companies: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving companies")

# ========================================
# AGENT MANAGEMENT
# ========================================

@router.post("/agents")
async def create_agent_with_documents(
    name: str = Form(...),
    type: str = Form(...),
    company_id: str = Form(...),
    prompt: str = Form(...),
    is_active: bool = Form(True),
    additional_context: Optional[str] = Form(None),  # JSON string
    advanced_settings: Optional[str] = Form(None),  # JSON string
    file_urls: Optional[str] = Form(None),  # JSON string of file URLs
    user_id: str = Form(...),  # User ID for the agent
    id: Optional[str] = Form(None),  # Accept existing ID to prevent duplication
    db: Session = Depends(get_db)
):
    """Create a new agent with support for multiple file URLs (documents and images)"""
    try:
        logger.info(f"Creating agent with name: {name}, type: {type}, company_id: {company_id}")
        
        # Check if company_id is "undefined" or not a valid UUID
        if company_id == "undefined" or not _is_valid_uuid(company_id):
            raise HTTPException(status_code=400, detail="Invalid company_id. A valid company ID is required.")
        
        # Verify company exists in the database
        company = db.query(Company).filter_by(id=company_id).first()
        if not company:
            raise HTTPException(status_code=404, detail=f"Company with ID {company_id} not found")
        
        # Parse additional_context if provided
        additional_context_dict = {}
        if additional_context:
            try:
                additional_context_dict = json.loads(additional_context)
            except json.JSONDecodeError:
                logger.warning("Invalid additional_context JSON provided")
        
        # Parse advanced_settings if provided
        advanced_settings_dict = {}
        if advanced_settings:
            try:
                advanced_settings_dict = json.loads(advanced_settings)
            except json.JSONDecodeError:
                logger.warning("Invalid advanced_settings JSON provided")
        
        # Use provided ID or generate a new one
        agent_id = id if id else str(uuid.uuid4())
        logger.info(f"Using agent ID: {agent_id} ({'provided' if id else 'generated'})")
        
        # Create agent with fields that match the database schema
        agent = Agent(
            id=agent_id,
            user_id=user_id,
            name=name,
            type=AgentType(type.lower()),
            company_id=company_id,
            prompt=prompt,
            additional_context=additional_context_dict,
            advanced_settings=advanced_settings_dict,
            is_active=is_active,
            files=[],  # Will be updated with file URLs
            # Fields with default values
            knowledge_base_ids=[],
            database_integration_ids=[],
            search_config={
                'score_threshold': 0.7,
                'limit': 5,
                'include_metadata': True
            },
            confidence_threshold=0.7,
            max_response_tokens=200,
            temperature=0.7,
            total_interactions=0,
            average_confidence=0.0,
            success_rate=0.0,
            average_response_time=0.0,
            image_processing_enabled=False,
            image_processing_config={
                'max_images': 1000,
                'confidence_threshold': 0.7,
                'enable_auto_description': True
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(agent)
        db.commit()
        db.refresh(agent)

        document_ids = []
        image_ids = []
        files = []  # List to store file URLs for agent.files field

        if file_urls:
            # Parse file URLs
            try:
                urls_list = json.loads(file_urls)
                logger.info(f"File URLs received: {urls_list}")
                files = urls_list
            except json.JSONDecodeError:
                logger.error("Invalid file URLs JSON provided")
                raise HTTPException(status_code=400, detail="Invalid file URLs format")

            # Initialize services
            qdrant_service = QdrantService()
            document_embedding_service = DocumentEmbeddingService(qdrant_service)
            image_embedding_service = ImageEmbeddingService(qdrant_service)
            
            # Ensure collection exists
            await qdrant_service.setup_collection(company_id)
            
            # Process each file URL
            image_files = []
            document_files = []
            
            for url in urls_list:
                try:
                    # Download file content
                    content, content_type = _download_from_s3_direct(url)
                    
                    if not content:
                        logger.warning(f"Failed to download content for URL: {url}")
                        continue
                    
                    filename = url.split('/')[-1]
                    logger.info(f"Downloaded file: {filename}, size: {len(content)} bytes, content type: {content_type}")
                    
                    # Determine if file is an image
                    is_image = _is_image_file(content_type)
                    
                    # Create document record with proper content handling
                    if is_image:
                        document = Document(
                            company_id=company_id,
                            agent_id=agent.id,
                            name=filename,
                            content=f"[Image: {filename}]",  # Placeholder text for images
                            image_content=content,  # Binary data
                            is_image=True,
                            file_size=len(content),
                            original_filename=filename,
                            file_type=content_type,
                            type=DocumentType.image,
                            image_metadata={
                                "original_filename": filename,
                                "file_url": url,
                                "uploaded_at": datetime.utcnow().isoformat()
                            }
                        )
                    else:
                        # For documents: decode content to text
                        try:
                            if isinstance(content, bytes):
                                text_content = content.decode('utf-8')
                            else:
                                text_content = str(content)
                        except UnicodeDecodeError:
                            text_content = content.decode('utf-8', errors='ignore')
                        
                        document = Document(
                            company_id=company_id,
                            agent_id=agent.id,
                            name=filename,
                            content=text_content,  # Decoded text
                            is_image=False,
                            file_size=len(content) if isinstance(content, bytes) else len(content.encode('utf-8')),
                            original_filename=filename,
                            file_type=content_type,
                            type=DocumentType.custom
                        )
                    
                    db.add(document)
                    db.commit()
                    db.refresh(document)
                    
                    # Add to appropriate list based on file type
                    if is_image:
                        image_files.append({
                            'id': document.id,
                            'content': content,
                            'filename': filename,
                            'content_type': content_type,
                            'metadata': {
                                'agent_id': agent.id,
                                'original_filename': filename,
                                'file_url': url
                            }
                        })
                        image_ids.append(document.id)
                    else:
                        document_files.append({
                            'id': document.id,
                            'content': content,
                            'metadata': {
                                'agent_id': agent.id,
                                'filename': filename,
                                'file_type': content_type,
                                'file_url': url
                            }
                        })
                        document_ids.append(document.id)
                            
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {str(e)}")
                    continue
            
            # Update agent with files field
            agent.files = files
            db.commit()
            
            # Embed documents if any
            if document_files:
                success = await document_embedding_service.embed_documents(
                    company_id=company_id,
                    agent_id=agent.id,
                    documents=document_files
                )
                if not success:
                    logger.error("Failed to embed documents")
            
            # Embed images if any
            if image_files:
                success = await image_embedding_service.embed_images(
                    company_id=company_id,
                    agent_id=agent.id,
                    images=image_files
                )
                if not success:
                    logger.error("Failed to embed images")

        return {
            "status": "success",
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type.value,
                "company_id": agent.company_id,
                "prompt": agent.prompt,
                "is_active": agent.is_active,
                "additional_context": agent.additional_context,
                "advanced_settings": agent.advanced_settings,
                "files": agent.files
            },
            "documents": {
                "total": len(document_ids) + len(image_ids),
                "document_ids": document_ids,
                "image_ids": image_ids
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating agent: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{company_id}")
async def get_company_agents(
    company_id: str, 
    db: Session = Depends(get_db)
):
    """Get all agents for a company"""
    try:
        agents = db.query(Agent).filter_by(company_id=company_id, is_active=True).all()
        return {
            "agents": [{
                "id": agent.id,
                "name": agent.name,
                "type": agent.type.value,
                "prompt": agent.prompt,
                "documents": len(agent.documents),
                "files": agent.files,
                "created_at": agent.created_at.isoformat(),
                "updated_at": agent.updated_at.isoformat()
            } for agent in agents],
            "total_agents": len(agents)
        }
    except Exception as e:
        logger.error(f"Error getting agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# DOCUMENT MANAGEMENT
# ========================================

@router.post("/documents/upload")
async def upload_documents(
    company_id: str = Form(...),
    agent_id: str = Form(...),
    files: List[UploadFile] = File(...),
    descriptions: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload and process documents and images for an existing agent"""
    try:
        # Check if agent exists
        agent = db.query(Agent).filter_by(id=agent_id, company_id=company_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Parse descriptions if provided
        descriptions_dict = {}
        if descriptions:
            try:
                descriptions_dict = json.loads(descriptions)
            except json.JSONDecodeError:
                logger.warning("Invalid descriptions JSON provided")
        
        # Initialize services
        qdrant_service = QdrantService()
        document_embedding_service = DocumentEmbeddingService(qdrant_service)
        image_embedding_service = ImageEmbeddingService(qdrant_service)
        
        # Ensure collection exists
        await qdrant_service.setup_collection(company_id)
        
        # Process files
        document_ids = []
        image_ids = []
        document_files = []
        image_files = []
        
        for file in files:
            try:
                content = await file.read()
                
                # Determine if file is an image
                is_image = _is_image_file(file.content_type)
                
                # Create document record
                if is_image:
                    document = Document(
                        company_id=company_id,
                        agent_id=agent_id,
                        name=file.filename,
                        content=f"[Image: {file.filename}]",
                        image_content=content,
                        is_image=True,
                        file_size=len(content),
                        original_filename=file.filename,
                        file_type=file.content_type,
                        type=DocumentType.image,
                        user_description=descriptions_dict.get(file.filename),
                        image_metadata={
                            "original_filename": file.filename,
                            "uploaded_at": datetime.utcnow().isoformat()
                        }
                    )
                else:
                    # Decode text content
                    try:
                        if isinstance(content, bytes):
                            text_content = content.decode('utf-8')
                        else:
                            text_content = str(content)
                    except UnicodeDecodeError:
                        text_content = content.decode('utf-8', errors='ignore')
                    
                    document = Document(
                        company_id=company_id,
                        agent_id=agent_id,
                        name=file.filename,
                        content=text_content,
                        is_image=False,
                        file_size=len(content),
                        original_filename=file.filename,
                        file_type=file.content_type,
                        type=DocumentType.custom
                    )
                
                db.add(document)
                db.commit()
                db.refresh(document)
                
                # Add to appropriate list based on file type
                if is_image:
                    image_files.append({
                        'id': document.id,
                        'content': content,
                        'description': descriptions_dict.get(file.filename),
                        'filename': file.filename,
                        'content_type': file.content_type,
                        'metadata': {
                            'agent_id': agent_id,
                            'original_filename': file.filename
                        }
                    })
                    image_ids.append(document.id)
                else:
                    document_files.append({
                        'id': document.id,
                        'content': content,
                        'metadata': {
                            'agent_id': agent_id,
                            'filename': file.filename,
                            'file_type': file.content_type
                        }
                    })
                    document_ids.append(document.id)
                        
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                continue
        
        # Process documents if any
        if document_files:
            success = await document_embedding_service.embed_documents(
                company_id=company_id,
                agent_id=agent_id,
                documents=document_files
            )
            if not success:
                logger.error("Failed to embed documents")
        
        # Process images if any
        if image_files:
            success = await image_embedding_service.embed_images(
                company_id=company_id,
                agent_id=agent_id,
                images=image_files
            )
            if not success:
                logger.error("Failed to embed images")
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {len(document_ids) + len(image_ids)} files",
            "documents": {
                "total": len(document_ids) + len(image_ids),
                "document_ids": document_ids,
                "image_ids": image_ids
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# CLIENT CONNECTION MANAGEMENT (DI Integration)
# ========================================

@router.get("/clients")
@inject
async def get_all_clients(
    limit: int = Query(100, description="Maximum number of clients to return"),
    offset: int = Query(0, description="Number of clients to skip"),
    company_filter: Optional[str] = Query(None, description="Filter by company ID"),
    connection_service: IConnectionService = Depends(Provide[Container.connection_service])
) -> List[Dict[str, Any]]:
    """Get all connected clients summary with pagination and filtering"""
    try:
        client_ids = connection_service.get_active_clients()
        clients_summary = []
        
        # Apply company filter if provided
        filtered_clients = []
        for client_id in client_ids:
            session = connection_service.get_client_session(client_id)
            if session:
                if company_filter and session.company:
                    if session.company.get("id") != company_filter:
                        continue
                filtered_clients.append((client_id, session))
        
        # Apply pagination
        paginated_clients = filtered_clients[offset:offset + limit]
        
        for client_id, session in paginated_clients:
            # Handle different session structures gracefully
            voice_active = False
            if hasattr(session, 'voice_state'):
                voice_active = session.voice_state != "inactive"
            elif hasattr(session, 'is_voice_call'):
                voice_active = session.is_voice_call
            
            clients_summary.append({
                "client_id": session.client_id,
                "company_name": session.company["name"] if session.company else "Unknown",
                "company_id": session.company["id"] if session.company else None,
                "connection_time": session.connection_time.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "message_count": session.message_count,
                "is_voice_call": voice_active,
                "session_duration": session.get_session_duration(),
                "total_tokens": getattr(session, 'total_tokens', 0),
                "initialized": getattr(session, 'initialized', False),
                "agent_id": getattr(session, 'agent_id', None)
            })
        
        return clients_summary
        
    except Exception as e:
        logger.error(f"Error getting clients: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving client list")

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
        
        # Handle voice duration safely
        voice_duration = 0.0
        if hasattr(session, 'get_voice_duration'):
            voice_duration = session.get_voice_duration()
        elif hasattr(session, 'voice_start_time') and session.voice_start_time:
            voice_duration = (datetime.utcnow() - session.voice_start_time).total_seconds()
        
        return {
            "client_id": session.client_id,
            "company": session.company,
            "connection_time": session.connection_time.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "session_duration": session.get_session_duration(),
            "message_count": session.message_count,
            "total_tokens": getattr(session, 'total_tokens', 0),
            "is_voice_call": getattr(session, 'is_voice_call', False),
            "voice_duration": voice_duration,
            "initialized": getattr(session, 'initialized', False),
            "agent_id": getattr(session, 'agent_id', None),
            "conversation_id": getattr(session, 'conversation_id', None),
            "request_times": getattr(session, 'request_times', []),
            "rate_limit_remaining": getattr(session, 'max_requests_per_minute', 60) - len(getattr(session, 'request_times', []))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting client details for {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving client details")

@router.delete("/clients/{client_id}")
@inject
async def force_disconnect_client(
    client_id: str,
    reason: str = Query("Admin disconnect", description="Reason for disconnection"),
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
            "timestamp": datetime.utcnow().isoformat(),
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error force disconnecting client {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Error disconnecting client")

# ========================================
# SYSTEM HEALTH AND ANALYTICS (DI Integration)
# ========================================

@router.get("/health")
@inject
async def system_health_check(
    detailed: bool = Query(False, description="Include detailed health information"),
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
        
        # Basic health response
        health_data = {
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
        
        # Add detailed information if requested
        if detailed:
            health_data.update({
                "memory_usage_mb": getattr(stats, 'memory_usage_mb', 0.0),
                "uptime_seconds": (datetime.utcnow() - stats.timestamp).total_seconds(),
                "peak_connections": getattr(stats, 'peak_connections', len(active_clients)),
                "rate_limited_clients": getattr(stats, 'rate_limited_clients', 0)
            })
        
        return health_data
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "total_connections": 0,
            "processing_active": 0,
            "warnings": [f"Health check failed: {str(e)}"],
            "error": str(e)
        }

@router.get("/analytics/company/{company_id}")
@inject
async def get_company_analytics(
    company_id: str,
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    metrics: Optional[str] = Query(None, description="Comma-separated list of metrics"),
    analytics_service: IAnalyticsService = Depends(Provide[Container.analytics_service])
) -> Dict[str, Any]:
    """Get analytics report for company"""
    try:
        # Parse metrics filter if provided
        metrics_filter = None
        if metrics:
            metrics_filter = [m.strip() for m in metrics.split(",")]
        
        report = await analytics_service.get_company_usage_report(
            company_id, start_date, end_date
        )
        
        # Apply metrics filter if specified
        if metrics_filter and isinstance(report, dict):
            filtered_report = {
                "company_id": report.get("company_id"),
                "period": report.get("period"),
                "totals": {},
                "daily_breakdown": report.get("daily_breakdown", [])
            }
            
            # Filter totals
            if "totals" in report:
                for metric in metrics_filter:
                    if metric in report["totals"]:
                        filtered_report["totals"][metric] = report["totals"][metric]
            
            return filtered_report
        
        return report
        
    except Exception as e:
        logger.error(f"Error getting company analytics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving analytics")

@router.get("/analytics/live")
@inject
async def get_live_analytics(
    analytics_service: IAnalyticsService = Depends(Provide[Container.analytics_service])
) -> Dict[str, Any]:
    """Get real-time system analytics"""
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
            "memory_usage_mb": getattr(stats, 'memory_usage_mb', 0.0),
            "rate_limiting": {
                "clients_limited": getattr(stats, 'rate_limited_clients', 0)
            }
        }
    except Exception as e:
        logger.error(f"Error getting live analytics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving live analytics")

@router.post("/broadcast")
@inject
async def broadcast_message(
    message: Dict[str, Any] = Body(..., description="Message to broadcast"),
    company_id: Optional[str] = Query(None, description="Optional company filter"),
    exclude_client: Optional[str] = Query(None, description="Client to exclude"),
    connection_service: IConnectionService = Depends(Provide[Container.connection_service])
) -> Dict[str, Any]:
    """Broadcast message to connected clients"""
    try:
        # Validate message format
        if not isinstance(message, dict) or "type" not in message:
            raise HTTPException(status_code=400, detail="Message must be a dict with 'type' field")
        
        # Add timestamp to message
        message["timestamp"] = datetime.utcnow().isoformat()
        message["source"] = "admin_broadcast"
        
        # Get active clients for counting
        active_clients = connection_service.get_active_clients()
        
        # Filter clients if company_id is specified
        target_count = len(active_clients)
        if company_id:
            target_count = 0
            for client_id in active_clients:
                session = connection_service.get_client_session(client_id)
                if session and session.company and session.company.get("id") == company_id:
                    target_count += 1
        
        # Exclude specific client if specified
        if exclude_client and exclude_client in active_clients:
            target_count -= 1
        
        # This would require implementing broadcast functionality in connection service
        # For now, return a placeholder response
        return {
            "message": "Broadcast queued",
            "target_clients": target_count,
            "company_filter": company_id,
            "excluded_client": exclude_client,
            "timestamp": datetime.utcnow().isoformat(),
            "broadcast_content": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        raise HTTPException(status_code=500, detail="Error broadcasting message")

@router.post("/maintenance")
async def trigger_maintenance(
    operation: str = Query(..., description="Maintenance operation to perform"),
    force: bool = Query(False, description="Force operation even if risky")
) -> Dict[str, Any]:
    """Trigger maintenance operations"""
    try:
        allowed_operations = ["cleanup_stale", "refresh_cache", "restart_services", "clear_logs"]
        
        if operation not in allowed_operations:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid operation. Allowed: {', '.join(allowed_operations)}"
            )
        
        # Placeholder for maintenance operations
        result = {
            "operation": operation,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "forced": force,
            "details": f"Maintenance operation '{operation}' completed successfully"
        }
        
        if operation == "cleanup_stale":
            result["details"] = "Cleaned up stale connections and cache entries"
        elif operation == "refresh_cache":
            result["details"] = "Refreshed all application caches"
        elif operation == "restart_services":
            result["details"] = "Restarted background services"
        elif operation == "clear_logs":
            result["details"] = "Cleared old log entries"
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing maintenance operation {operation}: {e}")
        raise HTTPException(status_code=500, detail="Error performing maintenance")

@router.get("/logs")
async def get_system_logs(
    level: str = Query("INFO", description="Log level filter"),
    limit: int = Query(100, description="Maximum number of log entries"),
    since: Optional[str] = Query(None, description="ISO timestamp to filter logs since")
) -> Dict[str, Any]:
    """Get system logs (placeholder for log aggregation)"""
    try:
        # This would integrate with your logging system
        # For now, return a placeholder response
        
        logs = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "module": "admin_routes",
                "message": "Log endpoint accessed",
                "details": {"level_filter": level, "limit": limit}
            }
        ]
        
        return {
            "logs": logs,
            "total_count": len(logs),
            "level_filter": level,
            "limit": limit,
            "since": since,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving system logs")

# ========================================
# HELPER FUNCTIONS
# ========================================

def _is_valid_uuid(val) -> bool:
    """Check if string is a valid UUID"""
    try:
        uuid_obj = uuid.UUID(str(val))
        return str(uuid_obj) == val
    except (ValueError, AttributeError, TypeError):
        return False

def _is_image_file(content_type: str) -> bool:
    """Check if the file is an image based on content type"""
    return content_type.startswith('image/')

def _list_s3_bucket_objects(bucket_name: str, prefix: str = '') -> List[str]:
    """List objects in an S3 bucket with an optional prefix"""
    try:
        import boto3
        logger.info(f"AWS Access Key ID present: {'Yes' if os.environ.get('aws_access_key_id') else 'No'}")
        logger.info(f"AWS Secret Access Key present: {'Yes' if os.environ.get('aws_secret_access_key') else 'No'}")
        s3_client = boto3.client(
            's3',
            region_name='ap-south-1',
            aws_access_key_id=os.environ.get("aws_access_key_id"),
            aws_secret_access_key=os.environ.get("aws_secret_access_key")
        )
        
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=100
        )
        
        if 'Contents' in response:
            objects = [obj['Key'] for obj in response['Contents']]
            logger.info(f"Found {len(objects)} objects in bucket {bucket_name} with prefix '{prefix}'")
            logger.info(f"Objects: {objects[:10]}")
            return objects
        else:
            logger.warning(f"No objects found in bucket {bucket_name} with prefix '{prefix}'")
            return []
    except Exception as e:
        logger.error(f"Error listing S3 bucket objects: {str(e)}")
        return []

def _download_from_s3_direct(url: str) -> tuple:
    """Download an S3 object directly using boto3, with special handling for URL formats"""
    # Input validation
    if not url or not isinstance(url, str):
        logger.error("Invalid URL provided")
        return None, None
    
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        logger.info(f"Attempting to download directly from URL: {url}")
        
        # Parse URL to get bucket and key
        parsed_url = urlparse(url)
        bucket_name, key = None, None
        
        # Check for different S3 URL formats
        if not parsed_url.netloc.endswith('.amazonaws.com'):
            logger.error(f"URL is not an S3 URL: {url}")
            return None, None
            
        # Format: https://bucket-name.s3.region.amazonaws.com/key
        if '.s3.' in parsed_url.netloc:
            parts = parsed_url.netloc.split('.s3.')
            bucket_name = parts[0]
            key = unquote(parsed_url.path.lstrip('/'))
        # Format: https://s3.region.amazonaws.com/bucket-name/key
        elif parsed_url.netloc.startswith('s3.'):
            path_parts = parsed_url.path.lstrip('/').split('/', 1)
            if len(path_parts) >= 2:
                bucket_name = path_parts[0]
                key = unquote(path_parts[1])
            else:
                logger.error(f"Invalid S3 URL format (path): {url}")
                return None, None
        else:
            logger.error(f"Unrecognized S3 URL format: {url}")
            return None, None
        
        # Validate parsed values
        if not bucket_name or not key:
            logger.error(f"Failed to parse bucket/key from URL: {url}")
            return None, None
            
        logger.info(f"Parsed S3 URL: bucket={bucket_name}, key={key}")
        
        # Check AWS credentials
        aws_access_key = os.environ.get("aws_access_key_id") or os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("aws_secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")
        
        if not aws_access_key or not aws_secret_key:
            logger.error("AWS credentials not found in environment variables")
            return None, None
        
        # Set up S3 client with proper error handling
        try:
            s3_client = boto3.client(
                's3',
                region_name='ap-south-1',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        except NoCredentialsError:
            logger.error("Invalid AWS credentials")
            return None, None
        except Exception as e:
            logger.error(f"Failed to create S3 client: {str(e)}")
            return None, None
        
        # Try to get the object directly first
        content, content_type = _try_download_object(s3_client, bucket_name, key)
        if content is not None:
            return content, content_type
        
        # If direct download failed, try with fallback strategies
        logger.info(f"Direct download failed, trying fallback strategies for: {key}")
        return _try_download_with_fallback(s3_client, bucket_name, key)
            
    except Exception as e:
        logger.error(f"Error in download_from_s3_direct: {str(e)}")
        return None, None

def _try_download_object(s3_client, bucket_name: str, key: str) -> tuple:
    """Try to download object directly"""
    try:
        logger.info(f"Getting object from S3: bucket={bucket_name}, key={key}")
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        content = response['Body'].read()
        content_type = response.get('ContentType', 'application/octet-stream')
        
        # Validate content
        if not content:
            logger.warning(f"Downloaded empty content for key: {key}")
            return None, None
            
        logger.info(f"Successfully downloaded S3 object: {key} ({len(content)} bytes)")
        return content, content_type
        
    except Exception as e:
        logger.error(f"Error downloading S3 object {key}: {str(e)}")
        return None, None

def _try_download_with_fallback(s3_client, bucket_name: str, original_key: str) -> tuple:
    """Try to download using fallback strategies"""
    try:
        # Strategy 1: Check if objects exist with similar names
        similar_objects = _list_s3_bucket_objects(bucket_name, original_key.split('/')[-1].split('-')[0])
        
        # Strategy 2: If no similar objects and key has dashes, try base name
        if not similar_objects and '-' in original_key:
            base_name = original_key.split('/')[-1].split('-')[0]
            similar_objects = _list_s3_bucket_objects(bucket_name, base_name)
        
        # Try downloading similar objects
        for similar_key in similar_objects[:5]:  # Limit to first 5 matches
            logger.info(f"Trying alternative key: {similar_key}")
            content, content_type = _try_download_object(s3_client, bucket_name, similar_key)
            if content is not None:
                return content, content_type
        
        logger.warning(f"No downloadable alternatives found for: {original_key}")
        return None, None
        
    except Exception as e:
        logger.error(f"Error in fallback download: {str(e)}")
        return None, None

def _get_content_type_from_extension(extension: str) -> str:
    """Get MIME type from file extension"""
    extension_to_mime = {
        'pdf': 'application/pdf',
        'doc': 'application/msword',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'txt': 'text/plain',
        'csv': 'text/csv',
        'xls': 'application/vnd.ms-excel',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'ppt': 'application/vnd.ms-powerpoint',
        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'svg': 'image/svg+xml',
        'html': 'text/html',
        'json': 'application/json'
    }
    
    return extension_to_mime.get(extension, 'application/octet-stream')