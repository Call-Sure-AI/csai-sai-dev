# src/application/dto/responses.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"

class BaseResponse(BaseModel):
    """Base response DTO"""
    status: ResponseStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message: Optional[str] = None

class ConnectionResponse(BaseResponse):
    """Response DTO for connection operations"""
    client_id: str
    company_name: Optional[str] = None
    agent_id: Optional[str] = None

class MessageResponse(BaseResponse):
    """Response DTO for message processing"""
    response_id: str
    content: str
    tokens_used: int
    response_time: float

class StreamChunkResponse(BaseModel):
    """Response DTO for streaming chunks"""
    type: str = "stream_chunk"
    content: str
    chunk_number: int
    msg_id: str
    is_final: bool = False

class VoiceResponse(BaseResponse):
    """Response DTO for voice operations"""
    session_id: str
    duration: Optional[float] = None
    audio_data: Optional[str] = None  # Base64 encoded

class ClientSummaryResponse(BaseModel):
    """Response DTO for client summary"""
    client_id: str
    company_name: str
    connection_time: datetime
    last_activity: datetime
    message_count: int
    is_voice_call: bool
    session_duration: float

class StatsResponse(BaseModel):
    """Response DTO for system statistics"""
    timestamp: datetime
    total_connections: int
    voice_calls_active: int
    companies_active: int
    processing_utilization: float

class HealthResponse(BaseModel):
    """Response DTO for health checks"""
    status: str
    timestamp: datetime
    total_connections: int
    processing_active: int
    warnings: List[str] = []

class AnalyticsResponse(BaseModel):
    """Response DTO for analytics data"""
    company_id: str
    period: Dict[str, Any]
    totals: Dict[str, Any]
    daily_breakdown: List[Dict[str, Any]]

class ErrorResponse(BaseResponse):
    """Response DTO for errors"""
    error_code: str
    details: Optional[Dict[str, Any]] = None
    
    def __init__(self, error_code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status=ResponseStatus.ERROR,
            message=message,
            error_code=error_code,
            details=details
        )