import asyncio
import json
from typing import Optional, AsyncGenerator
from sqlalchemy.orm import Session
from autogen import ConversableAgent
from autogen.agentchat import initiate_group_chat
from src.services.ag2.agent_builder import AG2AgentBuilder
from src.services.ag2.session_service import AG2SessionService
from src.services.agent_service import get_agent
from src.core.exceptions import AgentNotFoundError, InternalServerError
from src.utils.logger import setup_logger
from src.utils.otel import get_tracer

logger = setup_logger(__name__)


async def run_agent(
    agent_id: str,
    external_id: str,
    message: str,
    session_service: AG2SessionService,
    db: Session,
    session_id: Optional[str] = None,
    files: Optional[list] = None,
) -> dict:
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "ag2_run_agent",
        attributes={"agent_id": agent_id, "external_id": external_id},
    ):
        db_agent = get_agent(db, agent_id)
        if db_agent is None:
            raise AgentNotFoundError(f"Agent {agent_id} not found")

        builder = AG2AgentBuilder(db)
        result, _ = await builder.build_agent(db_agent)

        # Reconstruct conversation history as AG2 message list
        session = session_service.get_or_create(agent_id, external_id, session_id)
        history = session_service.build_messages(session)

        try:
            ag2_mode = (db_agent.config or {}).get("ag2_mode", "single")
            if ag2_mode == "group_chat":
                chat_result, final_context, last_agent = initiate_group_chat(
                    pattern=result["pattern"],
                    messages=history + [{"role": "user", "content": message}],
                    max_rounds=result["max_rounds"],
                    context_variables=result["context_variables"],
                )
                final_response = chat_result.summary or (
                    chat_result.chat_history[-1].get("content", "")
                    if chat_result.chat_history else "No response."
                )
                message_history = chat_result.chat_history

            else:
                # Single ConversableAgent — two-agent pattern with a silent proxy
                proxy = ConversableAgent(
                    name="user_proxy",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=1,
                    is_termination_msg=lambda x: True,  # one exchange only
                    llm_config=False,
                )
                # Run in executor to avoid blocking the event loop (AG2 is sync)
                loop = asyncio.get_event_loop()
                chat_result = await loop.run_in_executor(
                    None,
                    lambda: proxy.initiate_chat(
                        result,
                        message=message,
                        chat_history=history,
                        max_turns=1,
                    ),
                )
                final_response = (
                    chat_result.chat_history[-1].get("content", "")
                    if chat_result.chat_history else "No response."
                )
                message_history = chat_result.chat_history

            session_service.append(session, "user", message)
            session_service.append(session, "assistant", final_response)
            session_service.save(session)

            return {
                "final_response": final_response,
                "message_history": message_history,
            }

        except Exception as e:
            logger.error(f"AG2 runner error: {e}", exc_info=True)
            raise InternalServerError(str(e))


async def run_agent_stream(
    agent_id: str,
    external_id: str,
    message: str,
    session_service: AG2SessionService,
    db: Session,
    session_id: Optional[str] = None,
    files: Optional[list] = None,
) -> AsyncGenerator[str, None]:
    """
    AG2 does not provide token-level streaming in the way ADK does.
    We run the full exchange and yield the result as a single event chunk,
    matching the shape expected by the WebSocket handler in chat_routes.py.

    Token-level streaming can be added in a future iteration by wiring
    ConversableAgent's `process_last_received_message` hook to a queue.
    """
    result = await run_agent(
        agent_id=agent_id,
        external_id=external_id,
        message=message,
        session_service=session_service,
        db=db,
        session_id=session_id,
        files=files,
    )

    # Yield in the same event envelope shape as the ADK streaming runner
    yield json.dumps({
        "content": {
            "role": "agent",
            "parts": [{"type": "text", "text": result["final_response"]}],
        },
        "author": agent_id,
        "is_final": True,
    })
