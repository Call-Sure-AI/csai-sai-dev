# Codebase Modularization Plan

## Overview
This plan addresses the modularization of a FastAPI-based AI backend system that handles voice calls, WebRTC connections, and AI-powered conversations.

## Current Issues Identified
1. **Circular imports** - Services depend on each other creating import loops
2. **Large monolithic modules** - ConnectionManager has 600+ lines with multiple responsibilities
3. **Mixed concerns** - Business logic, data access, and presentation mixed together
4. **Tight coupling** - Components directly instantiate dependencies
5. **Configuration scattered** - Settings spread across multiple files
6. **No clear service boundaries** - Unclear separation between core services

## Proposed Module Structure

### 1. Core Domain Layer (`src/core/`)
```
src/core/
├── __init__.py
├── entities/           # Business entities (data classes)
│   ├── __init__.py
│   ├── client.py      # ClientSession, ConnectionState
│   ├── conversation.py # Conversation domain logic
│   ├── agent.py       # Agent domain logic
│   └── analytics.py   # Analytics domain entities
├── interfaces/         # Abstract interfaces/protocols
│   ├── __init__.py
│   ├── repositories.py # Data access interfaces
│   ├── services.py    # Service interfaces
│   └── events.py      # Event interfaces
└── exceptions.py      # Domain-specific exceptions
```

### 2. Infrastructure Layer (`src/infrastructure/`)
```
src/infrastructure/
├── __init__.py
├── database/
│   ├── __init__.py
│   ├── repositories/  # Concrete repository implementations
│   ├── models/        # SQLAlchemy models
│   └── migrations/    # Database migrations
├── external/          # External service integrations
│   ├── __init__.py
│   ├── twilio.py
│   ├── exotel.py
│   ├── deepgram.py
│   └── openai.py
├── storage/
│   ├── __init__.py
│   ├── s3.py
│   └── redis.py
└── messaging/         # Event/message handling
    ├── __init__.py
    └── events.py
```

### 3. Application Services (`src/application/`)
```
src/application/
├── __init__.py
├── services/          # Application services (use cases)
│   ├── __init__.py
│   ├── connection_service.py
│   ├── conversation_service.py
│   ├── voice_service.py
│   └── analytics_service.py
├── dto/              # Data Transfer Objects
│   ├── __init__.py
│   ├── requests.py   # Request DTOs
│   └── responses.py  # Response DTOs
└── handlers/         # Command/Query handlers
    ├── __init__.py
    ├── connection_handlers.py
    └── message_handlers.py
```

### 4. Dependency Injection (`src/di/`)
```
src/di/
├── __init__.py
├── container.py      # DI container
├── providers.py      # Service providers
└── registry.py       # Service registry
```

### 5. Presentation Layer (Keep existing `src/routes/`)
- Slim controllers that delegate to application services
- Input validation and response formatting only

## Key Modularization Changes

### 1. Extract ClientSession to Core Entity
**File: `src/core/entities/client.py`**
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from fastapi import WebSocket

@dataclass
class ClientSession:
    """Core client session entity with business logic"""
    client_id: str
    websocket: WebSocket
    connected: bool = True
    initialized: bool = False
    connection_time: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    # Session data
    company: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    agent_id: Optional[str] = None
    
    # Metrics
    message_count: int = 0
    total_tokens: int = 0
    request_times: List[float] = field(default_factory=list)
    
    # Voice state
    is_voice_call: bool = False
    voice_start_time: Optional[datetime] = None
    voice_callback: Optional[Callable] = None
    
    def update_activity(self, tokens: int = 0) -> None:
        """Update session activity metrics"""
        self.last_activity = datetime.utcnow()
        self.message_count += 1
        self.total_tokens += tokens
    
    def get_session_duration(self) -> float:
        """Calculate session duration in seconds"""
        return (datetime.utcnow() - self.connection_time).total_seconds()
```

### 2. Define Service Interfaces
**File: `src/core/interfaces/services.py`**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from .entities.client import ClientSession

class IConnectionService(ABC):
    @abstractmethod
    async def connect_client(self, client_id: str, websocket: WebSocket) -> bool:
        pass
    
    @abstractmethod
    async def disconnect_client(self, client_id: str) -> None:
        pass
    
    @abstractmethod
    async def authenticate_client(self, client_id: str, api_key: str) -> bool:
        pass

class IConversationService(ABC):
    @abstractmethod
    async def process_message(self, client_id: str, message: str) -> str:
        pass
    
    @abstractmethod
    async def get_conversation_context(self, conversation_id: str) -> List[Dict]:
        pass

class IVoiceService(ABC):
    @abstractmethod
    async def start_voice_call(self, client_id: str) -> bool:
        pass
    
    @abstractmethod
    async def process_audio_chunk(self, client_id: str, audio_data: bytes) -> None:
        pass
```

### 3. Implement Dependency Injection
**File: `src/di/container.py`**
```python
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide

from application.services.connection_service import ConnectionService
from application.services.conversation_service import ConversationService
from infrastructure.database.repositories.client_repository import ClientRepository
from infrastructure.external.deepgram import DeepgramService

class Container(containers.DeclarativeContainer):
    # Configuration
    config = providers.Configuration()
    
    # Database
    database = providers.Singleton(
        lambda: get_database_session()
    )
    
    # Repositories
    client_repository = providers.Factory(
        ClientRepository,
        session=database
    )
    
    # External services
    deepgram_service = providers.Singleton(
        DeepgramService,
        api_key=config.deepgram.api_key
    )
    
    # Application services
    connection_service = providers.Factory(
        ConnectionService,
        client_repository=client_repository
    )
    
    conversation_service = providers.Factory(
        ConversationService,
        client_repository=client_repository
    )
```

### 4. Refactor ConnectionManager to Service
**File: `src/application/services/connection_service.py`**
```python
from typing import Dict, Optional
from fastapi import WebSocket
from core.interfaces.services import IConnectionService
from core.entities.client import ClientSession
from core.interfaces.repositories import IClientRepository

class ConnectionService(IConnectionService):
    def __init__(self, 
                 client_repository: IClientRepository,
                 max_connections: int = 1000):
        self.clients: Dict[str, ClientSession] = {}
        self.client_repository = client_repository
        self.max_connections = max_connections
    
    async def connect_client(self, client_id: str, websocket: WebSocket) -> bool:
        if len(self.clients) >= self.max_connections:
            await websocket.close(code=1013, reason="Server at capacity")
            return False
        
        await websocket.accept()
        session = ClientSession(client_id, websocket)
        self.clients[client_id] = session
        
        # Persist connection event
        await self.client_repository.record_connection(session)
        return True
    
    async def disconnect_client(self, client_id: str) -> None:
        session = self.clients.get(client_id)
        if not session:
            return
        
        # Persist disconnection event
        await self.client_repository.record_disconnection(session)
        
        # Cleanup
        if not session.is_websocket_closed():
            await session.websocket.close()
        
        self.clients.pop(client_id, None)
    
    async def authenticate_client(self, client_id: str, api_key: str) -> bool:
        session = self.clients.get(client_id)
        if not session:
            return False
        
        # Authenticate via repository
        company = await self.client_repository.authenticate_company(api_key)
        if not company:
            return False
        
        session.company = company
        return True
```

### 5. Create Repository Layer
**File: `src/infrastructure/database/repositories/client_repository.py`**
```python
from sqlalchemy.orm import Session
from core.interfaces.repositories import IClientRepository
from core.entities.client import ClientSession
from database.models import Company, SessionEvent

class ClientRepository(IClientRepository):
    def __init__(self, session: Session):
        self.session = session
    
    async def authenticate_company(self, api_key: str) -> Optional[Dict]:
        company = self.session.query(Company).filter_by(api_key=api_key).first()
        if not company:
            return None
        
        return {
            "id": company.id,
            "name": company.name,
            "api_key": company.api_key
        }
    
    async def record_connection(self, session: ClientSession) -> None:
        if not session.company:
            return
        
        event = SessionEvent(
            id=f"{session.client_id}_{int(time.time())}",
            company_id=session.company["id"],
            client_id=session.client_id,
            event_type="connect",
            timestamp=session.connection_time
        )
        self.session.add(event)
        self.session.commit()
```

### 6. Slim Down Route Handlers
**File: `src/routes/webrtc_handlers.py`**
```python
from fastapi import APIRouter, WebSocket, Depends
from dependency_injector.wiring import Provide, inject
from di.container import Container
from core.interfaces.services import IConnectionService, IConversationService

router = APIRouter()

@router.websocket("/signal/{peer_id}/{company_api_key}")
@inject
async def websocket_endpoint(
    websocket: WebSocket,
    peer_id: str,
    company_api_key: str,
    connection_service: IConnectionService = Depends(Provide[Container.connection_service]),
    conversation_service: IConversationService = Depends(Provide[Container.conversation_service])
):
    # Simple delegation to services
    connected = await connection_service.connect_client(peer_id, websocket)
    if not connected:
        return
    
    authenticated = await connection_service.authenticate_client(peer_id, company_api_key)
    if not authenticated:
        await connection_service.disconnect_client(peer_id)
        return
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                response = await conversation_service.process_message(
                    peer_id, 
                    data.get("message", "")
                )
                await websocket.send_json({"type": "response", "content": response})
                
    except WebSocketDisconnect:
        await connection_service.disconnect_client(peer_id)
```

### 7. Configuration Management
**File: `src/config/settings_manager.py`**
```python
from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    # Database
    database_url: str
    
    # External APIs
    deepgram_api_key: str
    openai_api_key: str
    twilio_account_sid: str
    twilio_auth_token: str
    
    # Application
    max_connections: int = 1000
    max_requests_per_minute: int = 60
    
    class Config:
        env_file = ".env"

class SettingsManager:
    _instance = None
    _settings = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = Settings()
        return self._settings
    
    def get_config_dict(self) -> Dict[str, Any]:
        return self.settings.dict()

# Global instance
settings_manager = SettingsManager()
```

## Implementation Steps

### Phase 1: Foundation (Week 1)
1. Create core module structure
2. Extract domain entities (ClientSession, etc.)
3. Define service interfaces
4. Set up dependency injection container

### Phase 2: Service Layer (Week 2)
1. Implement ConnectionService
2. Implement ConversationService
3. Implement VoiceService
4. Create repository layer

### Phase 3: Infrastructure (Week 3)
1. Refactor external service integrations
2. Implement concrete repositories
3. Set up event messaging system
4. Update configuration management

### Phase 4: Presentation (Week 4)
1. Refactor route handlers to use DI
2. Implement DTOs for requests/responses
3. Add proper error handling
4. Update tests

### Phase 5: Analytics & Cleanup (Week 5)
1. Refactor analytics service
2. Implement proper logging
3. Performance optimization
4. Documentation updates

## Benefits of This Approach

1. **Separation of Concerns**: Each module has a single responsibility
2. **Testability**: Easy to mock dependencies and unit test
3. **Maintainability**: Clear boundaries and minimal coupling
4. **Scalability**: Easy to add new features without affecting existing code
5. **Flexibility**: Can swap implementations without changing business logic
6. **Clean Architecture**: Follows SOLID principles and clean architecture patterns

## Migration Strategy

1. **Gradual Migration**: Start with one service at a time
2. **Backward Compatibility**: Keep existing endpoints working during migration
3. **Feature Flags**: Use feature flags to toggle between old/new implementations
4. **Parallel Implementation**: Implement new services alongside existing ones
5. **Comprehensive Testing**: Ensure all functionality works before removing old code

This modularization will transform the current monolithic structure into a maintainable, scalable, and testable architecture.




The modularization plan I've provided transforms your monolithic FastAPI backend into a clean, maintainable architecture. Here's a summary of the key benefits and implementation approach:
Key Benefits of This Modularization:
1. Separation of Concerns

Core Domain: Business logic isolated from infrastructure
Application Services: Use cases and workflows
Infrastructure: Database, external APIs, messaging
Presentation: Slim controllers focused on HTTP concerns

2. Dependency Inversion

Services depend on interfaces, not concrete implementations
Easy to swap implementations (e.g., switch from Deepgram to another STT service)
Simplified testing with mock implementations

3. Clean Architecture Benefits

Testability: Each layer can be tested independently
Maintainability: Clear boundaries and responsibilities
Scalability: Easy to add new features without affecting existing code
Flexibility: Technology choices can change without affecting business logic

4. Professional Structure

Industry-standard patterns (Repository, Service Layer, DI)
Clear naming conventions and file organization
Comprehensive error handling and logging
Configuration management

Implementation Strategy:
Phase 1: Foundation (Week 1)

Create the core module structure
Extract ClientSession and other entities
Define service interfaces
Set up basic DI container

Phase 2: Service Layer (Week 2)

Implement ConnectionService, ConversationService
Create repository implementations
Refactor existing WebSocket handlers

Phase 3: Infrastructure (Week 3)

Implement external service wrappers
Database repository layer
Event messaging system

Phase 4: Integration (Week 4)

Wire everything together with DI
Update route handlers
Comprehensive testing

Phase 5: Optimization (Week 5)

Performance tuning
Analytics and monitoring
Documentation and deployment

Migration Tips:

Gradual Migration: Start with one service at a time
Backward Compatibility: Keep existing endpoints working
Feature Flags: Toggle between old/new implementations
Comprehensive Testing: Ensure functionality during migration

This architecture will make your codebase much more maintainable, testable, and scalable while following industry best practices. The dependency injection system ensures loose coupling, and the clean separation of concerns makes it easy to understand and modify individual components.