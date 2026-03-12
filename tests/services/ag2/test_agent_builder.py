"""Unit tests for AG2AgentBuilder — no LLM calls, no DB, no network."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.services.ag2.agent_builder import AG2AgentBuilder

@pytest.fixture
def mock_db():
    return AsyncMock()

def _make_agent(agent_id, agent_name, ag2_mode="single", model="gpt-4o-mini"):
    """Create a minimal agent record mock with explicit attribute assignment."""
    a = MagicMock()
    a.id = agent_id
    a.name = agent_name
    a.type = "llm"
    a.llm_provider = "openai"
    a.model = model
    a.config = {"ag2_mode": ag2_mode, "api_key": "test-key"}
    a.api_key = "test-key"
    a.api_key_id = None
    a.api_url = "https://api.openai.com/v1"
    a.role = None
    a.goal = None
    a.instruction = f"You are {agent_name}."
    return a

@pytest.fixture
def group_chat_agent_record():
    """Group-chat root agent with two sub-agents."""
    a = _make_agent("agent-123", "Support Team")
    a.config = {
        "ag2_mode": "group_chat",
        "sub_agents": ["uuid-triage", "uuid-specialist"],
        "max_rounds": 10,
        "pattern": "auto",
        "api_key": "test-key",
    }
    return a

@pytest.mark.asyncio
async def test_builder_creates_group_chat(mock_db, group_chat_agent_record):
    builder = AG2AgentBuilder(db=mock_db)
    sub_agent = _make_agent("uuid-triage", "triage")
    with patch("src.services.ag2.agent_builder.get_agent", return_value=sub_agent):
        result, _ = await builder.build_agent(group_chat_agent_record)
    # result should be a dict with pattern and agents for initiate_group_chat
    assert "pattern" in result
    assert "agents" in result

@pytest.mark.asyncio
async def test_builder_validates_group_chat_requires_agents(mock_db):
    record = MagicMock()
    record.type = "llm"
    record.config = {"ag2_mode": "group_chat", "sub_agents": []}  # empty — should raise
    record.api_key = "test"
    builder = AG2AgentBuilder(db=mock_db)
    with pytest.raises(ValueError, match="at least one"):
        await builder.build_agent(record)
