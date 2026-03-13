"""Unit tests for AG2AgentRunner streaming interface."""
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from src.services.ag2.agent_runner import run_agent_stream, run_agent


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
