"""Unit tests for AG2AgentRunner streaming interface."""
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from src.services.ag2.agent_runner import run_agent_stream, run_agent
from src.core.exceptions import AgentNotFoundError, InternalServerError


@pytest.mark.asyncio
async def test_runner_yields_response_chunks():
    mock_result = {
        "final_response": "Resolved: your issue is fixed.",
        "message_history": [],
    }
    mock_session_service = MagicMock()

    with patch("src.services.ag2.agent_runner.run_agent", AsyncMock(return_value=mock_result)):
        chunks = []
        async for chunk in run_agent_stream(
            db=AsyncMock(),
            agent_id="agent-123",
            external_id="ext-123",
            session_service=mock_session_service,
            session_id="session-abc",
            message="My printer is broken",
        ):
            chunks.append(chunk)

    assert len(chunks) >= 1
    data = json.loads(chunks[0])
    # Validate full streaming envelope contract
    assert data["content"]["role"] == "agent"
    assert data["author"] == "agent-123"
    assert data["content"]["parts"][0]["type"] == "text"
    assert data["content"]["parts"][0]["text"] == "Resolved: your issue is fixed."
    assert data["is_final"] is True


@pytest.mark.asyncio
async def test_run_agent_persists_session_history_in_order():
    """append is called user-then-assistant, save is called once."""
    mock_db = MagicMock()
    mock_session_service = MagicMock()
    mock_session = MagicMock()
    mock_session_service.get_or_create.return_value = mock_session
    mock_session_service.build_messages.return_value = []

    user_message = "My printer is broken"

    mock_agent_record = MagicMock()
    mock_agent_record.config = {"ag2_mode": "single"}

    mock_chat_result = MagicMock()
    mock_chat_result.chat_history = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "Try turning it off and on again."},
    ]

    with patch("src.services.ag2.agent_runner.get_agent", return_value=mock_agent_record), \
         patch("src.services.ag2.agent_runner.AG2AgentBuilder") as MockBuilder, \
         patch("src.services.ag2.agent_runner.ConversableAgent") as MockProxy, \
         patch("src.services.ag2.agent_runner.asyncio") as mock_asyncio:

        builder_instance = MockBuilder.return_value
        builder_instance.build_agent = AsyncMock(return_value=(MagicMock(), None))

        proxy_instance = MockProxy.return_value
        mock_asyncio.get_event_loop.return_value.run_in_executor = AsyncMock(
            return_value=mock_chat_result
        )

        result = await run_agent(
            agent_id="agent-123",
            external_id="ext-123",
            message=user_message,
            session_service=mock_session_service,
            db=mock_db,
            session_id="session-abc",
        )

    final_response = result["final_response"]
    mock_session_service.append.assert_has_calls([
        call(mock_session, "user", user_message),
        call(mock_session, "assistant", final_response),
    ])
    mock_session_service.save.assert_called_once_with(mock_session)


@pytest.mark.asyncio
async def test_run_agent_raises_agent_not_found():
    """run_agent raises AgentNotFoundError when the agent DB record is missing."""
    with patch("src.services.ag2.agent_runner.get_agent", return_value=None):
        with pytest.raises(AgentNotFoundError):
            await run_agent(
                agent_id="missing-agent",
                external_id="ext-123",
                message="hello",
                session_service=MagicMock(),
                db=AsyncMock(),
            )


@pytest.mark.asyncio
async def test_run_agent_stream_propagates_exception():
    """run_agent_stream propagates exceptions raised by run_agent."""
    with patch(
        "src.services.ag2.agent_runner.run_agent",
        AsyncMock(side_effect=InternalServerError("boom")),
    ):
        with pytest.raises(InternalServerError):
            async for _ in run_agent_stream(
                agent_id="agent-123",
                external_id="ext-123",
                message="hello",
                session_service=MagicMock(),
                db=AsyncMock(),
            ):
                pass


@pytest.mark.asyncio
async def test_run_agent_group_chat_failure_raises_internal_error():
    """Exceptions from initiate_group_chat are wrapped and re-raised as InternalServerError."""
    mock_agent_record = MagicMock()
    mock_agent_record.config = {"ag2_mode": "group_chat"}

    mock_session_service = MagicMock()
    mock_session_service.get_or_create.return_value = MagicMock()
    mock_session_service.build_messages.return_value = []

    group_chat_setup = {
        "pattern": MagicMock(),
        "max_rounds": 10,
        "context_variables": MagicMock(),
    }

    with patch("src.services.ag2.agent_runner.get_agent", return_value=mock_agent_record), \
         patch("src.services.ag2.agent_runner.AG2AgentBuilder") as MockBuilder, \
         patch(
             "src.services.ag2.agent_runner.initiate_group_chat",
             side_effect=RuntimeError("llm failure"),
         ):
        MockBuilder.return_value.build_agent = AsyncMock(return_value=(group_chat_setup, None))

        with pytest.raises(InternalServerError, match="llm failure"):
            await run_agent(
                agent_id="agent-123",
                external_id="ext-123",
                message="hello",
                session_service=mock_session_service,
                db=AsyncMock(),
            )
