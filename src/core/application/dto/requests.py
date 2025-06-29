# src/application/dto/requests.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class MessageRequest(BaseModel):
    """Request DTO for processing messages"""
    message: str = Field(..., min_length=1, max_length=5000)
    metadata: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None

class VoiceStartRequest(BaseModel):
    """Request DTO for starting voice calls"""
    voice_settings: Optional[Dict[str, Any]] = None
    language: str = Field(default="en-US")
    enable_interim_results: bool = Field(default=True)

class AudioChunkRequest(BaseModel):
    """Request DTO for audio chunks"""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    sequence_number: int = Field(..., ge=0)
    is_final: bool = Field(default=False)

class ClientAuthRequest(BaseModel):
    """Request DTO for client authentication"""
    api_key: str = Field(..., min_length=10)
    client_info: Optional[Dict[str, Any]] = None

class AgentInitRequest(BaseModel):
    """Request DTO for agent initialization"""
    agent_id: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None

class AnalyticsQueryRequest(BaseModel):
    """Request DTO for analytics queries"""
    company_id: str
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    metrics: Optional[List[str]] = None