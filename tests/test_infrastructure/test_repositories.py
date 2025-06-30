# tests/test_infrastructure/test_repositories.py
"""
Tests for infrastructure repositories.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.orm import Session

from infrastructure.database.repositories.client_repository import ClientRepository
from infrastructure.database.repositories.conversation_repository import ConversationRepository
from infrastructure.database.models import Company, SessionEvent, EventType
from core.entities.conversation import Conversation, Message, MessageType


class TestClientRepository:
    """Tests for ClientRepository."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock SQLAlchemy session."""
        session = Mock(spec=Session)
        session.query.return_value = session
        session.filter.return_value = session
        session.filter_by.return_value = session
        session.first.return_value = None
        session.commit.return_value = None
        session.rollback.return_value = None
        session.add.return_value = None
        return session
    
    @pytest.fixture
    def client_repository(self, mock_session):
        """Create ClientRepository instance."""
        return ClientRepository(mock_session)
    
    @pytest.mark.asyncio
    async def test_authenticate_company_success(self, client_repository, mock_session):
        """Test successful company authentication."""
        # Setup mock company
        mock_company = Mock()
        mock_company.id = "company_1"
        mock_company.name = "Test Company"
        mock_company.api_key = "test_api_key"
        mock_company.max_connections = 100
        mock_company.max_requests_per_minute = 60
        mock_company.metadata = {"test": "data"}
        
        mock_session.first.return_value = mock_company
        
        # Test authentication
        result = await client_repository.authenticate_company("test_api_key")
        
        # Assertions
        assert result is not None
        assert result["id"] == "company_1"
        assert result["name"] == "Test Company"
        assert result["api_key"] == "test_api_key"
        assert result["max_connections"] == 100
        assert result["max_requests_per_minute"] == 60
        assert result["metadata"] == {"test": "data"}
    
    @pytest.mark.asyncio
    async def test_authenticate_company_not_found(self, client_repository, mock_session):
        """Test company authentication when company not found."""
        mock_session.first.return_value = None
        
        result = await client_repository.authenticate_company("invalid_key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_record_connection_event(self, client_repository, mock_session):
        """Test recording connection event."""
        session_id = "test_session_123"
        event_type = "connection"
        metadata = {"test": "metadata"}
        
        await client_repository.record_connection_event(session_id, event_type, metadata)
        
        # Verify session.add was called
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_session_count_by_company(self, client_repository, mock_session):
        """Test getting session count for company."""
        mock_session.count.return_value = 5
        
        count = await client_repository.get_session_count_by_company("company_1")
        
        assert count == 5


class TestConversationRepository:
    """Tests for ConversationRepository."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock SQLAlchemy session."""
        session = Mock(spec=Session)
        session.query.return_value = session
        session.filter.return_value = session
        session.filter_by.return_value = session
        session.first.return_value = None
        session.all.return_value = []
        session.commit.return_value = None
        session.rollback.return_value = None
        session.add.return_value = None
        return session
    
    @pytest.fixture
    def conversation_repository(self, mock_session):
        """Create ConversationRepository instance."""
        return ConversationRepository(mock_session)
    
    @pytest.mark.asyncio
    async def test_save_conversation_new(self, conversation_repository, mock_session):
        """Test saving a new conversation."""
        from core.entities.conversation import ConversationState
        
        conversation = Conversation(
            id="conv_123",
            client_id="client_123",
            agent_id="agent_123"
        )
        conversation.state = ConversationState.ACTIVE
        
        mock_session.first.return_value = None  # Conversation doesn't exist
        
        await conversation_repository.save_conversation(conversation)