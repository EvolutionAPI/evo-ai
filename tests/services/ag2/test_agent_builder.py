"""Unit tests for AG2AgentBuilder — no LLM calls, no DB, no network."""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from autogen import ConversableAgent
from autogen.agentchat.group import TerminateTarget, RevertToUserTarget
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
    a.description = ""
    return a


@pytest.fixture
def group_chat_agent_record():
    """Group-chat root agent with two sub-agents."""
    a = _make_agent("agent-123", "Support_Team")
    a.config = {
        "ag2_mode": "group_chat",
        "sub_agents": ["uuid-triage", "uuid-specialist"],
        "max_rounds": 10,
        "pattern": "auto",
        "api_key": "test-key",
    }
    return a


# ---------------------------------------------------------------------------
# Group-chat mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_builder_creates_group_chat(mock_db, group_chat_agent_record):
    builder = AG2AgentBuilder(db=mock_db)
    sub_agent = _make_agent("uuid-triage", "triage")
    with patch("src.services.ag2.agent_builder.get_agent", return_value=sub_agent):
        result, _ = await builder.build_agent(group_chat_agent_record)
    assert "pattern" in result
    assert "agents" in result


@pytest.mark.asyncio
async def test_builder_validates_group_chat_requires_agents(mock_db):
    record = MagicMock()
    record.type = "llm"
    record.config = {"ag2_mode": "group_chat", "sub_agents": []}
    record.api_key = "test"
    builder = AG2AgentBuilder(db=mock_db)
    with pytest.raises(ValueError, match="at least one"):
        await builder.build_agent(record)


# ---------------------------------------------------------------------------
# Single-agent mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_builder_single_mode_returns_conversable_agent(mock_db):
    """build_agent with ag2_mode=single returns a ConversableAgent, not a dict."""
    record = _make_agent("agent-abc", "Support_Bot", ag2_mode="single")
    builder = AG2AgentBuilder(db=mock_db)
    agent, meta = await builder.build_agent(record)
    assert isinstance(agent, ConversableAgent)
    assert meta is None


@pytest.mark.asyncio
async def test_builder_default_mode_is_single(mock_db):
    """Omitting ag2_mode in config defaults to single-agent mode."""
    record = _make_agent("agent-abc", "My_Bot")
    record.config = {"api_key": "test-key"}  # no ag2_mode key
    builder = AG2AgentBuilder(db=mock_db)
    agent, _ = await builder.build_agent(record)
    assert isinstance(agent, ConversableAgent)


@pytest.mark.asyncio
async def test_builder_sanitizes_agent_name(mock_db):
    """Spaces in agent names are replaced with underscores for AG2 compatibility."""
    record = _make_agent("agent-abc", "Support Team Agent")
    builder = AG2AgentBuilder(db=mock_db)
    agent, _ = await builder.build_agent(record)
    assert " " not in agent.name
    assert agent.name == "Support_Team_Agent"


@pytest.mark.asyncio
async def test_builder_system_message_includes_role_goal_instruction(mock_db):
    """System message is composed from role, goal, and instruction fields."""
    record = _make_agent("agent-abc", "Bot")
    record.role = "billing specialist"
    record.goal = "resolve billing disputes"
    record.instruction = "Always be concise."
    builder = AG2AgentBuilder(db=mock_db)
    agent, _ = await builder.build_agent(record)
    sm = agent.system_message
    assert "billing specialist" in sm
    assert "resolve billing disputes" in sm
    assert "Always be concise." in sm


# ---------------------------------------------------------------------------
# _get_api_key branches
# ---------------------------------------------------------------------------

def test_get_api_key_uses_api_key_id(mock_db):
    """api_key_id takes priority and resolves via get_decrypted_api_key."""
    record = _make_agent("agent-abc", "Bot")
    record.api_key_id = "11111111-2222-3333-4444-555555555555"
    record.config = {}
    builder = AG2AgentBuilder(db=mock_db)
    with patch(
        "src.services.ag2.agent_builder.get_decrypted_api_key",
        return_value="resolved-key",
    ):
        key = asyncio.run(builder._get_api_key(record))
    assert key == "resolved-key"


def test_get_api_key_falls_back_to_raw_string(mock_db):
    """Non-UUID api_key in config is returned as-is."""
    record = _make_agent("agent-abc", "Bot")
    record.api_key_id = None
    record.config = {"api_key": "raw-openai-key"}
    builder = AG2AgentBuilder(db=mock_db)
    key = asyncio.run(builder._get_api_key(record))
    assert key == "raw-openai-key"


def test_get_api_key_raises_when_no_key_configured(mock_db):
    """ValueError raised when neither api_key_id nor config api_key is set."""
    record = _make_agent("agent-abc", "Bot")
    record.api_key_id = None
    record.config = {}
    builder = AG2AgentBuilder(db=mock_db)
    with pytest.raises(ValueError, match="No API key"):
        asyncio.run(builder._get_api_key(record))


# ---------------------------------------------------------------------------
# _apply_handoffs — validation
# ---------------------------------------------------------------------------

def test_apply_handoffs_skips_missing_type(mock_db):
    """Handoff entries with missing 'type' are skipped without raising."""
    builder = AG2AgentBuilder(db=mock_db)
    ca = MagicMock()
    # Should not raise even with no 'type' key
    builder._apply_handoffs(ca, {"handoffs": [{"target_agent_id": "x"}]}, {})


def test_apply_handoffs_skips_unknown_type(mock_db):
    """Handoff entries with unknown type value are skipped without raising."""
    builder = AG2AgentBuilder(db=mock_db)
    ca = MagicMock()
    builder._apply_handoffs(
        ca, {"handoffs": [{"type": "unknown", "target_agent_id": "x"}]}, {}
    )


def test_apply_handoffs_registers_llm_condition(mock_db):
    """LLM-type handoff adds an OnCondition via add_llm_conditions."""
    builder = AG2AgentBuilder(db=mock_db)
    ca = MagicMock()
    target = MagicMock()
    target.name = "target_agent"  # AgentTarget validates agent_name as str
    all_agents = {"target-uuid": target}
    config = {
        "handoffs": [
            {"type": "llm", "target_agent_id": "target-uuid", "condition": "user asks about billing"}
        ]
    }
    builder._apply_handoffs(ca, config, all_agents)
    ca.handoffs.add_llm_conditions.assert_called_once()
    conditions = ca.handoffs.add_llm_conditions.call_args[0][0]
    assert len(conditions) == 1


def test_apply_handoffs_registers_context_condition(mock_db):
    """Context-type handoff adds an OnContextCondition via add_context_conditions."""
    builder = AG2AgentBuilder(db=mock_db)
    ca = MagicMock()
    target = MagicMock()
    target.name = "target_agent"  # AgentTarget validates agent_name as str
    all_agents = {"target-uuid": target}
    config = {
        "handoffs": [
            {"type": "context", "target_agent_id": "target-uuid", "expression": "${is_vip} == True"}
        ]
    }
    builder._apply_handoffs(ca, config, all_agents)
    ca.handoffs.add_context_conditions.assert_called_once()
    conditions = ca.handoffs.add_context_conditions.call_args[0][0]
    assert len(conditions) == 1


def test_apply_handoffs_after_work_terminate(mock_db):
    """after_work='terminate' sets TerminateTarget on the agent."""
    builder = AG2AgentBuilder(db=mock_db)
    ca = MagicMock()
    builder._apply_handoffs(ca, {"after_work": "terminate", "handoffs": []}, {})
    call_arg = ca.handoffs.set_after_work.call_args[0][0]
    assert isinstance(call_arg, TerminateTarget)


def test_apply_handoffs_after_work_default_reverts_to_user(mock_db):
    """Omitting after_work defaults to RevertToUserTarget."""
    builder = AG2AgentBuilder(db=mock_db)
    ca = MagicMock()
    builder._apply_handoffs(ca, {"handoffs": []}, {})
    call_arg = ca.handoffs.set_after_work.call_args[0][0]
    assert isinstance(call_arg, RevertToUserTarget)


@pytest.mark.asyncio
async def test_build_group_chat_setup_raises_on_missing_sub_agent(mock_db):
    """build_group_chat_setup raises ValueError when a sub-agent is not found in the DB."""
    record = MagicMock()
    record.config = {"ag2_mode": "group_chat", "sub_agents": ["missing-uuid"]}
    record.api_key = "test"
    builder = AG2AgentBuilder(db=mock_db)
    with patch("src.services.ag2.agent_builder.get_agent", return_value=None):
        with pytest.raises(ValueError, match="not found"):
            await builder.build_group_chat_setup(record)
