"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ @author: Davidson Gomes                                                      │
│ @file: chat_routes.py                                                        │
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

import uuid
import base64
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    WebSocket,
    WebSocketDisconnect,
    Header,
)
from sqlalchemy.orm import Session
from src.config.settings import settings
from src.config.database import get_db
from src.core.jwt_middleware import (
    get_jwt_token,
    verify_user_client,
    get_jwt_token_ws,
)
from src.services import (
    agent_service,
)
from src.schemas.chat import ChatRequest, ChatResponse, ErrorResponse, FileData
from src.services.adk.agent_runner import run_agent as run_agent_adk, run_agent_stream
from src.services.crewai.agent_runner import run_agent as run_agent_crewai
from src.services.ag2.agent_runner import run_agent as run_agent_ag2, run_agent_stream as run_agent_stream_ag2
from src.core.exceptions import AgentNotFoundError
from src.services.service_providers import (
    session_service,
    artifacts_service,
    memory_service,
)

from datetime import datetime
import logging
import json
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)


async def get_agent_by_api_key(
    agent_id: str,
    api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Flexible authentication for chat routes, allowing JWT or API key"""
    if authorization:
        # Try to authenticate with JWT token first
        try:
            # Extract token from Authorization header if needed
            token = (
                authorization.replace("Bearer ", "")
                if authorization.startswith("Bearer ")
                else authorization
            )
            payload = await get_jwt_token(token)
            agent = agent_service.get_agent(db, agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Agent not found",
                )

            # Verify if the user has access to the agent's client
            await verify_user_client(payload, db, agent.client_id)
            return agent
        except Exception as e:
            logger.warning(f"JWT authentication failed: {str(e)}")
            # If JWT fails, continue to try with API key

    # Try to authenticate with API key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required (JWT or API key)",
        )

    agent = agent_service.get_agent(db, agent_id)
    if not agent or not agent.config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )

    # Verify if the API key matches
    if not agent.config.get("api_key") or agent.config.get("api_key") != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    return agent


@router.websocket("/ws/{agent_id}/{external_id}")
async def websocket_chat(
    websocket: WebSocket,
    agent_id: str,
    external_id: str,
    db: Session = Depends(get_db),
):
    try:
        # Accept the connection
        await websocket.accept()
        logger.info("WebSocket connection accepted, waiting for authentication")

        # Wait for authentication message
        try:
            auth_data = await websocket.receive_json()
            logger.info(f"Authentication data received: {auth_data}")

            if not (
                auth_data.get("type") == "authorization"
                and (auth_data.get("token") or auth_data.get("api_key"))
            ):
                logger.warning("Invalid authentication message")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

            # Verify if the agent exists
            agent = agent_service.get_agent(db, agent_id)
            if not agent:
                logger.warning(f"Agent {agent_id} not found")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

            # Verify authentication
            is_authenticated = False

            # Try with JWT token
            if auth_data.get("token"):
                try:
                    payload = await get_jwt_token_ws(auth_data["token"])
                    if payload:
                        # Verify if the user has access to the agent
                        await verify_user_client(payload, db, agent.client_id)
                        is_authenticated = True
                except Exception as e:
                    logger.warning(f"JWT authentication failed: {str(e)}")

            # If JWT fails, try with API key
            if not is_authenticated and auth_data.get("api_key"):
                if agent.config and agent.config.get("api_key") == auth_data.get(
                    "api_key"
                ):
                    is_authenticated = True
                else:
                    logger.warning("Invalid API key")

            if not is_authenticated:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

            logger.info(
                f"WebSocket connection established for agent {agent_id} and external_id {external_id}"
            )

            while True:
                try:
                    data = await websocket.receive_json()
                    logger.info(f"Received message: {data}")
                    message = data.get("message")

                    if not message:
                        continue

                    files = None
                    if data.get("files") and isinstance(data.get("files"), list):
                        try:
                            files = []
                            for file_data in data.get("files"):
                                if (
                                    isinstance(file_data, dict)
                                    and file_data.get("filename")
                                    and file_data.get("content_type")
                                    and file_data.get("data")
                                ):
                                    files.append(
                                        FileData(
                                            filename=file_data.get("filename"),
                                            content_type=file_data.get("content_type"),
                                            data=file_data.get("data"),
                                        )
                                    )
                            logger.info(f"Processed {len(files)} files via WebSocket")
                        except Exception as e:
                            logger.error(f"Error processing files: {str(e)}")
                            files = None

                    if settings.AI_ENGINE == "ag2":
                        stream_gen = run_agent_stream_ag2(
                            agent_id=agent_id,
                            external_id=external_id,
                            message=message,
                            session_service=session_service,
                            db=db,
                            files=files,
                        )
                    else:
                        stream_gen = run_agent_stream(
                            agent_id=agent_id,
                            external_id=external_id,
                            message=message,
                            session_service=session_service,
                            artifacts_service=artifacts_service,
                            memory_service=memory_service,
                            db=db,
                            files=files,
                        )
                    async for chunk in stream_gen:
                        await websocket.send_json(
                            {"message": json.loads(chunk), "turn_complete": False}
                        )

                    # Send signal of complete turn
                    await websocket.send_json({"message": "", "turn_complete": True})

                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                    break
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON message received")
                    continue
                except Exception as e:
                    logger.error(f"Error in WebSocket message handling: {str(e)}")
                    await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                    break

        except WebSocketDisconnect:
            logger.info("Client disconnected during authentication")
        except json.JSONDecodeError:
            logger.warning("Invalid authentication message format")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        except Exception as e:
            logger.error(f"Error during authentication: {str(e)}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)


@router.post(
    "/{agent_id}/{external_id}",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def chat(
    request: ChatRequest,
    agent_id: str,
    external_id: str,
    _=Depends(get_agent_by_api_key),
    db: Session = Depends(get_db),
):
    try:
        if settings.AI_ENGINE == "adk":
            final_response = await run_agent_adk(
                agent_id,
                external_id,
                request.message,
                session_service,
                artifacts_service,
                memory_service,
                db,
                files=request.files,
            )
        elif settings.AI_ENGINE == "crewai":
            final_response = await run_agent_crewai(
                agent_id,
                external_id,
                request.message,
                session_service,
                db,
                files=request.files,
            )
        elif settings.AI_ENGINE == "ag2":
            final_response = await run_agent_ag2(
                agent_id,
                external_id,
                request.message,
                session_service,
                db,
                files=request.files,
            )

        return {
            "response": final_response["final_response"],
            "message_history": final_response["message_history"],
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        }

    except AgentNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e
