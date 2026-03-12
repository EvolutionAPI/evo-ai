"""Unit tests for AG2AgentRunner streaming interface."""
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.services.ag2.agent_runner import run_agent_stream

@pytest.mark.asyncio
async def test_runner_yields_response_chunks():
    mock_result = {
        "final_response": "Resolved: your issue is fixed.",
        "message_history": [],
    }
    mock_settings = MagicMock()
    mock_settings.POSTGRES_CONNECTION_STRING = "sqlite://"

    # Patch run_agent (the heavy work), session service, and settings —
    # get_settings and AG2SessionService are local imports inside run_agent_stream
    with patch("src.services.ag2.agent_runner.run_agent", AsyncMock(return_value=mock_result)):
        with patch("src.services.ag2.session_service.AG2SessionService"):
            with patch("src.config.settings.get_settings", return_value=mock_settings):
                chunks = []
                async for chunk in run_agent_stream(
                    db=AsyncMock(),
                    agent_id="agent-123",
                    external_id="ext-123",
                    session_id="session-abc",
                    message="My printer is broken",
                ):
                    chunks.append(chunk)

    assert len(chunks) >= 1
    data = json.loads(chunks[0])
    assert data["content"]["parts"][0]["text"] == "Resolved: your issue is fixed."
    assert data["is_final"] is True
