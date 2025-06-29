import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

# Mock the database models before they are imported by the manager
from src.database.models import Agent, Company, Conversation

# Path to the module being tested
AGENT_MANAGER_PATH = "src.managers.agent_manager"

from src.managers.agent_manager import AgentManager

# --- Pytest Fixtures ---

@pytest.fixture
def mock_db_session():
    """Creates a mock of the SQLAlchemy Session."""
    db_session = MagicMock()
    db_session.query.return_value.filter_by.return_value.first.return_value = None
    db_session.query.return_value.filter.return_value.all.return_value = []
    return db_session

@pytest.fixture
def mock_vector_store():
    """Creates a mock of the QdrantService vector store."""
    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=[])
    vector_store.get_query_embedding = AsyncMock(return_value=[0.1] * 1536)
    return vector_store

@pytest.fixture
def agent_manager(mock_db_session, mock_vector_store) -> AgentManager:
    """Initializes the AgentManager with mocked dependencies."""
    return AgentManager(mock_db_session, mock_vector_store)

# --- Test Cases ---

@pytest.mark.asyncio
async def test_ensure_base_agent_creation(agent_manager: AgentManager, mock_db_session):
    """Test that a base agent is created if one does not exist."""
    company_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    # Simulate that no base agent exists initially
    mock_db_session.query.return_value.filter_by.return_value.first.return_value = None
    
    # Simulate finding the company to get the user_id
    mock_company = Company(id=company_id, user_id=user_id)
    agent_manager.db.query(Company).filter_by.return_value.first.return_value = mock_company

    # Call the function
    base_agent = await agent_manager.ensure_base_agent(company_id)

    # Assertions
    assert base_agent is not None
    assert base_agent["name"] == "Base Agent"
    assert base_agent["type"] == "base"
    
    # Check that the agent was added to the database
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_find_best_agent_fallback(agent_manager: AgentManager, mock_vector_store):
    """Test that it falls back to the base agent when a vector search returns nothing."""
    company_id = str(uuid.uuid4())
    query = "I need help with my bill."
    
    # Simulate vector search returning no results
    mock_vector_store.search.return_value = []

    # Mock the get_base_agent to return a predictable base agent
    base_agent_info = {"id": "base-agent-id", "type": "base"}
    agent_manager.get_base_agent = AsyncMock(return_value=base_agent_info)

    # Call the function
    agent_id, score = await agent_manager.find_best_agent(company_id, query)

    # Assertions
    assert agent_id == "base-agent-id"
    assert score == 0.0
    # Assert that the mock was called (we don't care how many times, just that the fallback happened)
    assert agent_manager.get_base_agent.called