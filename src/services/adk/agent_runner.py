"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ @author: Davidson Gomes                                                      │
│ @file: agent_runner.py                                                       │
│ Developed by: Davidson Gomes                                                 │
│ Creation date: May 13, 2025                                                  │
│ Contact: contato@evolution-api.com                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ @copyright © Evolution API 2025. All rights reserved.                        │
│ Licensed under the Apache License, Version 2.0                               │
│                                                                              │
│ You may not use this file except in compliance with the License.             │
│ You may obtain a copy of the License at                                      │
│                                                                              │
│    http://www.apache.org/licenses/LICENSE-2.0                                │
│                                                                              │
│ Unless required by applicable law or agreed to in writing, software          │
│ distributed under the License is distributed on an "AS IS" BASIS,            │
│ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     │
│ See the License for the specific language governing permissions and          │
│ limitations under the License.                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ @important                                                                   │
│ For any future changes to the code in this file, it is recommended to        │
│ include, together with the modification, the information of the developer    │
│ who changed it and the date of modification.                                 │
└──────────────────────────────────────────────────────────────────────────────┘
"""

from google.adk.runners import Runner
from google.genai.types import Content, Part, Blob
from google.adk.sessions import DatabaseSessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from src.utils.logger import setup_logger
from src.core.exceptions import AgentNotFoundError, InternalServerError
from src.services.agent_service import get_agent
from src.services.adk.agent_builder import AgentBuilder
from sqlalchemy.orm import Session
from typing import Optional, AsyncGenerator
import asyncio
import json
from src.utils.otel import get_tracer
from opentelemetry import trace
import base64

logger = setup_logger(__name__)


async def run_agent(
    agent_id: str,
    external_id: str,
    message: str,
    session_service: DatabaseSessionService,
    artifacts_service: InMemoryArtifactService,
    memory_service: InMemoryMemoryService,
    db: Session,
    session_id: Optional[str] = None,
    timeout: float = 60.0,
    files: Optional[list] = None,
):
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "run_agent",
        attributes={
            "agent_id": agent_id,
            "external_id": external_id,
            "session_id": session_id or f"{external_id}_{agent_id}",
            "message": message,
            "has_files": files is not None and len(files) > 0,
        },
    ):
        exit_stack = None
        try:
            logger.info(
                f"Starting execution of agent {agent_id} for external_id {external_id}"
            )
            logger.info(f"Received message: {message}")

            if files and len(files) > 0:
                logger.info(f"Received {len(files)} files with message")

            get_root_agent = get_agent(db, agent_id)
            logger.info(
                f"Root agent found: {get_root_agent.name} (type: {get_root_agent.type})"
            )

            if get_root_agent is None:
                raise AgentNotFoundError(f"Agent with ID {agent_id} not found")

            # Using the AgentBuilder to create the agent
            agent_builder = AgentBuilder(db)
            root_agent, exit_stack = await agent_builder.build_agent(get_root_agent)

            logger.info("Configuring Runner")
            agent_runner = Runner(
                agent=root_agent,
                app_name=agent_id,
                session_service=session_service,
                artifact_service=artifacts_service,
                memory_service=memory_service,
            )
            adk_session_id = f"{external_id}_{agent_id}"
            if session_id is None:
                session_id = adk_session_id

            logger.info(f"Searching session for external_id {external_id}")
            session = session_service.get_session(
                app_name=agent_id,
                user_id=external_id,
                session_id=adk_session_id,
            )

            if session is None:
                logger.info(f"Creating new session for external_id {external_id}")
                session = session_service.create_session(
                    app_name=agent_id,
                    user_id=external_id,
                    session_id=adk_session_id,
                )

            file_parts = []
            if files and len(files) > 0:
                for file_data in files:
                    try:
                        file_bytes = base64.b64decode(file_data.data)

                        logger.info(f"DEBUG - Processing file: {file_data.filename}")
                        logger.info(f"DEBUG - File size: {len(file_bytes)} bytes")
                        logger.info(f"DEBUG - MIME type: '{file_data.content_type}'")
                        logger.info(f"DEBUG - First 20 bytes: {file_bytes[:20]}")

                        try:
                            file_part = Part(
                                inline_data=Blob(
                                    mime_type=file_data.content_type, data=file_bytes
                                )
                            )
                            logger.info(f"DEBUG - Part created successfully")
                        except Exception as part_error:
                            logger.error(
                                f"DEBUG - Error creating Part: {str(part_error)}"
                            )
                            logger.error(
                                f"DEBUG - Error type: {type(part_error).__name__}"
                            )
                            import traceback

                            logger.error(
                                f"DEBUG - Stack trace: {traceback.format_exc()}"
                            )
                            raise

                        # Save the file in the ArtifactService
                        version = artifacts_service.save_artifact(
                            app_name=agent_id,
                            user_id=external_id,
                            session_id=adk_session_id,
                            filename=file_data.filename,
                            artifact=file_part,
                        )
                        logger.info(
                            f"Saved file {file_data.filename} as version {version}"
                        )

                        # Add the Part to the list of parts for the message content
                        file_parts.append(file_part)
                    except Exception as e:
                        logger.error(
                            f"Error processing file {file_data.filename}: {str(e)}"
                        )

            # Create the content with the text message and the files
            parts = [Part(text=message)]
            if file_parts:
                parts.extend(file_parts)

            content = Content(role="user", parts=parts)
            logger.info("Starting agent execution")

            final_response_text = "No final response captured."
            message_history = []

            try:
                response_queue = asyncio.Queue()
                execution_completed = asyncio.Event()

                async def process_events():
                    try:
                        events_async = agent_runner.run_async(
                            user_id=external_id,
                            session_id=adk_session_id,
                            new_message=content,
                        )

                        last_response = None
                        all_responses = []

                        async for event in events_async:
                            if event.content and event.content.parts:
                                event_dict = event.dict()
                                event_dict = convert_sets(event_dict)
                                message_history.append(event_dict)

                            if (
                                event.content
                                and event.content.parts
                                and event.content.parts[0].text
                            ):
                                current_text = event.content.parts[0].text
                                last_response = current_text
                                all_responses.append(current_text)

                            if event.actions and event.actions.escalate:
                                escalate_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                                await response_queue.put(escalate_text)
                                execution_completed.set()
                                return

                        if last_response:
                            await response_queue.put(last_response)
                        else:
                            await response_queue.put(
                                "Finished without specific response"
                            )

                        execution_completed.set()
                    except Exception as e:
                        logger.error(f"Error in process_events: {str(e)}")
                        await response_queue.put(f"Error: {str(e)}")
                        execution_completed.set()

                task = asyncio.create_task(process_events())

                try:
                    wait_task = asyncio.create_task(execution_completed.wait())
                    done, pending = await asyncio.wait({wait_task}, timeout=timeout)

                    for p in pending:
                        p.cancel()

                    if not execution_completed.is_set():
                        logger.warning(
                            f"Agent execution timed out after {timeout} seconds"
                        )
                        await response_queue.put(
                            "The response took too long and was interrupted."
                        )

                    final_response_text = await response_queue.get()

                except Exception as e:
                    logger.error(f"Error waiting for response: {str(e)}")
                    final_response_text = f"Error processing response: {str(e)}"

                # Add the session to memory after completion
                completed_session = session_service.get_session(
                    app_name=agent_id,
                    user_id=external_id,
                    session_id=adk_session_id,
                )

                memory_service.add_session_to_memory(completed_session)

                # Cancel the processing task if it is still running
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        logger.info("Task cancelled successfully")
                    except Exception as e:
                        logger.error(f"Error cancelling task: {str(e)}")

            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                raise InternalServerError(str(e)) from e

            logger.info("Agent execution completed successfully")
            return {
                "final_response": final_response_text,
                "message_history": message_history,
            }
        except AgentNotFoundError as e:
            logger.error(f"Error processing request: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Internal error processing request: {str(e)}", exc_info=True)
            raise InternalServerError(str(e))
        finally:
            # Clean up MCP connection - MUST be executed in the same task
            if exit_stack:
                logger.info("Closing MCP server connection...")
                try:
                    await exit_stack.aclose()
                except Exception as e:
                    logger.error(f"Error closing MCP connection: {e}")
                    # Do not raise the exception to not obscure the original error


def convert_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets(i) for i in obj]
    else:
        return obj


async def run_agent_stream(
    agent_id: str,
    external_id: str,
    message: str,
    session_service: DatabaseSessionService,
    artifacts_service: InMemoryArtifactService,
    memory_service: InMemoryMemoryService,
    db: Session,
    session_id: Optional[str] = None,
    files: Optional[list] = None,
) -> AsyncGenerator[str, None]:
    tracer = get_tracer()
    span = tracer.start_span(
        "run_agent_stream",
        attributes={
            "agent_id": agent_id,
            "external_id": external_id,
            "session_id": session_id or f"{external_id}_{agent_id}",
            "message": message,
            "has_files": files is not None and len(files) > 0,
        },
    )
    try:
        with trace.use_span(span, end_on_exit=True):
            try:
                logger.info(
                    f"Starting streaming execution of agent {agent_id} for external_id {external_id}"
                )
                logger.info(f"Received message: {message}")

                if files and len(files) > 0:
                    logger.info(f"Received {len(files)} files with message")

                get_root_agent = get_agent(db, agent_id)
                logger.info(
                    f"Root agent found: {get_root_agent.name} (type: {get_root_agent.type})"
                )

                if get_root_agent is None:
                    raise AgentNotFoundError(f"Agent with ID {agent_id} not found")

                # Using the AgentBuilder to create the agent
                agent_builder = AgentBuilder(db)
                root_agent, exit_stack = await agent_builder.build_agent(get_root_agent)

                logger.info("Configuring Runner")
                agent_runner = Runner(
                    agent=root_agent,
                    app_name=agent_id,
                    session_service=session_service,
                    artifact_service=artifacts_service,
                    memory_service=memory_service,
                )
                adk_session_id = f"{external_id}_{agent_id}"
                if session_id is None:
                    session_id = adk_session_id

                logger.info(f"Searching session for external_id {external_id}")
                session = session_service.get_session(
                    app_name=agent_id,
                    user_id=external_id,
                    session_id=adk_session_id,
                )

                if session is None:
                    logger.info(f"Creating new session for external_id {external_id}")
                    session = session_service.create_session(
                        app_name=agent_id,
                        user_id=external_id,
                        session_id=adk_session_id,
                    )

                # Process the received files
                file_parts = []
                if files and len(files) > 0:
                    for file_data in files:
                        try:
                            # Decode the base64 file
                            file_bytes = base64.b64decode(file_data.data)

                            # Detailed debug
                            logger.info(
                                f"DEBUG - Processing file: {file_data.filename}"
                            )
                            logger.info(f"DEBUG - File size: {len(file_bytes)} bytes")
                            logger.info(
                                f"DEBUG - MIME type: '{file_data.content_type}'"
                            )
                            logger.info(f"DEBUG - First 20 bytes: {file_bytes[:20]}")

                            # Create a Part for the file using the default constructor
                            try:
                                file_part = Part(
                                    inline_data=Blob(
                                        mime_type=file_data.content_type,
                                        data=file_bytes,
                                    )
                                )
                                logger.info(f"DEBUG - Part created successfully")
                            except Exception as part_error:
                                logger.error(
                                    f"DEBUG - Error creating Part: {str(part_error)}"
                                )
                                logger.error(
                                    f"DEBUG - Error type: {type(part_error).__name__}"
                                )
                                import traceback

                                logger.error(
                                    f"DEBUG - Stack trace: {traceback.format_exc()}"
                                )
                                raise

                            # Save the file in the ArtifactService
                            version = artifacts_service.save_artifact(
                                app_name=agent_id,
                                user_id=external_id,
                                session_id=adk_session_id,
                                filename=file_data.filename,
                                artifact=file_part,
                            )
                            logger.info(
                                f"Saved file {file_data.filename} as version {version}"
                            )

                            # Add the Part to the list of parts for the message content
                            file_parts.append(file_part)
                        except Exception as e:
                            logger.error(
                                f"Error processing file {file_data.filename}: {str(e)}"
                            )

                # Create the content with the text message and the files
                parts = [Part(text=message)]
                if file_parts:
                    parts.extend(file_parts)

                content = Content(role="user", parts=parts)
                logger.info("Starting agent streaming execution")

                try:
                    events_async = agent_runner.run_async(
                        user_id=external_id,
                        session_id=adk_session_id,
                        new_message=content,
                    )

                    async for event in events_async:
                        try:
                            event_dict = event.model_dump()
                            event_dict = convert_sets(event_dict)

                            if "content" in event_dict and event_dict["content"]:
                                content = event_dict["content"]

                                if "role" not in content or content["role"] not in [
                                    "user",
                                    "agent",
                                ]:
                                    content["role"] = "agent"

                                if "parts" in content and content["parts"]:
                                    valid_parts = []
                                    for part in content["parts"]:
                                        if isinstance(part, dict):
                                            if "type" not in part and "text" in part:
                                                part["type"] = "text"
                                                valid_parts.append(part)
                                            elif "type" in part:
                                                valid_parts.append(part)

                                    if valid_parts:
                                        content["parts"] = valid_parts
                                    else:
                                        content["parts"] = [
                                            {
                                                "type": "text",
                                                "text": "Content without valid format",
                                            }
                                        ]
                                else:
                                    content["parts"] = [
                                        {
                                            "type": "text",
                                            "text": "Content without parts",
                                        }
                                    ]

                            # Send the individual event
                            yield json.dumps(event_dict)
                        except Exception as e:
                            logger.error(f"Error processing event: {e}")
                            continue

                    completed_session = session_service.get_session(
                        app_name=agent_id,
                        user_id=external_id,
                        session_id=adk_session_id,
                    )

                    memory_service.add_session_to_memory(completed_session)
                except Exception as e:
                    logger.error(f"Error processing request: {str(e)}")
                    raise InternalServerError(str(e)) from e
                finally:
                    # Clean up MCP connection
                    if exit_stack:
                        logger.info("Closing MCP server connection...")
                        try:
                            await exit_stack.aclose()
                        except Exception as e:
                            logger.error(f"Error closing MCP connection: {e}")

                logger.info("Agent streaming execution completed successfully")
            except AgentNotFoundError as e:
                logger.error(f"Error processing request: {str(e)}")
                raise InternalServerError(str(e)) from e
            except Exception as e:
                logger.error(
                    f"Internal error processing request: {str(e)}", exc_info=True
                )
                raise InternalServerError(str(e))
    finally:
        span.end()
