"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ @author: Davidson Gomes                                                      │
│ @file: session_routes.py                                                     │
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

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.config.database import get_db
from typing import List, Optional, Dict, Any
import uuid
import base64
from src.core.jwt_middleware import (
    get_jwt_token,
    verify_user_client,
)
from src.services import (
    agent_service,
)
from google.adk.events import Event
from google.adk.sessions import Session as Adk_Session
from src.services.session_service import (
    get_session_events,
    get_session_by_id,
    delete_session,
    get_sessions_by_agent,
    get_sessions_by_client,
)
from src.services.service_providers import session_service, artifacts_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/sessions",
    tags=["sessions"],
    responses={404: {"description": "Not found"}},
)


# Session Routes
@router.get("/client/{client_id}")
async def get_client_sessions(
    client_id: uuid.UUID,
    db: Session = Depends(get_db),
    payload: dict = Depends(get_jwt_token),
):
    # Verify if the user has access to this client's data
    await verify_user_client(payload, db, client_id)
    return get_sessions_by_client(db, client_id)


@router.get("/agent/{agent_id}")
async def get_agent_sessions(
    agent_id: uuid.UUID,
    db: Session = Depends(get_db),
    payload: dict = Depends(get_jwt_token),
    skip: int = 0,
    limit: int = 100,
):
    # Verify if the agent belongs to the user's client
    agent = agent_service.get_agent(db, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )

    # Verify if the user has access to the agent (via client)
    await verify_user_client(payload, db, agent.client_id)

    return get_sessions_by_agent(db, agent_id, skip, limit)


@router.get("/{session_id}", response_model=Adk_Session)
async def get_session(
    session_id: str,
    db: Session = Depends(get_db),
    payload: dict = Depends(get_jwt_token),
):
    # Get the session
    session = get_session_by_id(session_service, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    # Verify if the session's agent belongs to the user's client
    agent_id = uuid.UUID(session.app_name) if session.app_name else None
    if agent_id:
        agent = agent_service.get_agent(db, agent_id)
        if agent:
            await verify_user_client(payload, db, agent.client_id)

    return session


@router.get(
    "/{session_id}/messages",
)
async def get_agent_messages(
    session_id: str,
    db: Session = Depends(get_db),
    payload: dict = Depends(get_jwt_token),
):
    """
    Gets messages from a session with embedded artifacts.

    This function loads all messages from a session and processes any references
    to artifacts, loading them and converting them to base64 for direct use in the frontend.
    """
    # Get the session
    session = get_session_by_id(session_service, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    # Verify if the session's agent belongs to the user's client
    agent_id = uuid.UUID(session.app_name) if session.app_name else None
    if agent_id:
        agent = agent_service.get_agent(db, agent_id)
        if agent:
            await verify_user_client(payload, db, agent.client_id)

    # Parse session ID para obter app_name e user_id
    parts = session_id.split("_")
    if len(parts) != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session ID format"
        )

    user_id, app_name = parts[0], parts[1]

    events = get_session_events(session_service, session_id)

    processed_events = []
    for event in events:
        event_dict = event.model_dump()

        def process_dict(d):
            if isinstance(d, dict):
                for key, value in list(d.items()):
                    if isinstance(value, bytes):
                        try:
                            d[key] = base64.b64encode(value).decode("utf-8")
                            logger.debug(f"Converted bytes field to base64: {key}")
                        except Exception as e:
                            logger.error(f"Error encoding bytes to base64: {str(e)}")
                            d[key] = None
                    elif isinstance(value, dict):
                        process_dict(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, (dict, list)):
                                process_dict(item)
            elif isinstance(d, list):
                for i, item in enumerate(d):
                    if isinstance(item, bytes):
                        try:
                            d[i] = base64.b64encode(item).decode("utf-8")
                        except Exception as e:
                            logger.error(
                                f"Error encoding bytes to base64 in list: {str(e)}"
                            )
                            d[i] = None
                    elif isinstance(item, (dict, list)):
                        process_dict(item)
            return d

        # Process all event dictionary
        event_dict = process_dict(event_dict)

        # Process the content parts specifically
        if event_dict.get("content") and event_dict["content"].get("parts"):
            for part in event_dict["content"]["parts"]:
                # Process inlineData if present
                if part and part.get("inlineData") and part["inlineData"].get("data"):
                    # Check if it's already a string or if it's bytes
                    if isinstance(part["inlineData"]["data"], bytes):
                        # Convert bytes to base64 string
                        part["inlineData"]["data"] = base64.b64encode(
                            part["inlineData"]["data"]
                        ).decode("utf-8")
                        logger.debug(
                            f"Converted binary data to base64 in message {event_dict.get('id')}"
                        )

                # Process fileData if present (reference to an artifact)
                if part and part.get("fileData") and part["fileData"].get("fileId"):
                    try:
                        # Extract the file name from the fileId
                        file_id = part["fileData"]["fileId"]

                        # Load the artifact from the artifacts service
                        artifact = artifacts_service.load_artifact(
                            app_name=app_name,
                            user_id=user_id,
                            session_id=session_id,
                            filename=file_id,
                        )

                        if artifact and hasattr(artifact, "inline_data"):
                            # Extract the data and MIME type
                            file_bytes = artifact.inline_data.data
                            mime_type = artifact.inline_data.mime_type

                            # Add inlineData with the artifact data
                            if not part.get("inlineData"):
                                part["inlineData"] = {}

                            # Ensure we're sending a base64 string, not bytes
                            if isinstance(file_bytes, bytes):
                                try:
                                    part["inlineData"]["data"] = base64.b64encode(
                                        file_bytes
                                    ).decode("utf-8")
                                except Exception as e:
                                    logger.error(
                                        f"Error encoding artifact to base64: {str(e)}"
                                    )
                                    part["inlineData"]["data"] = None
                            else:
                                part["inlineData"]["data"] = str(file_bytes)

                            part["inlineData"]["mimeType"] = mime_type

                            logger.debug(
                                f"Loaded artifact {file_id} for message {event_dict.get('id')}"
                            )
                    except Exception as e:
                        logger.error(f"Error loading artifact: {str(e)}")
                        # Don't interrupt the flow if an artifact fails

        # Check artifact_delta in actions
        if event_dict.get("actions") and event_dict["actions"].get("artifact_delta"):
            artifact_deltas = event_dict["actions"]["artifact_delta"]
            for filename, version in artifact_deltas.items():
                try:
                    # Load the artifact
                    artifact = artifacts_service.load_artifact(
                        app_name=app_name,
                        user_id=user_id,
                        session_id=session_id,
                        filename=filename,
                        version=version,
                    )

                    if artifact and hasattr(artifact, "inline_data"):
                        # If the event doesn't have an artifacts section, create it
                        if "artifacts" not in event_dict:
                            event_dict["artifacts"] = {}

                        # Add the artifact to the event's artifacts list
                        file_bytes = artifact.inline_data.data
                        mime_type = artifact.inline_data.mime_type

                        # Ensure the bytes are converted to base64
                        event_dict["artifacts"][filename] = {
                            "data": (
                                base64.b64encode(file_bytes).decode("utf-8")
                                if isinstance(file_bytes, bytes)
                                else str(file_bytes)
                            ),
                            "mimeType": mime_type,
                            "version": version,
                        }

                        logger.debug(
                            f"Added artifact {filename} (v{version}) to message {event_dict.get('id')}"
                        )
                except Exception as e:
                    logger.error(
                        f"Error processing artifact_delta {filename}: {str(e)}"
                    )

        processed_events.append(event_dict)

    return processed_events


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_session(
    session_id: str,
    db: Session = Depends(get_db),
    payload: dict = Depends(get_jwt_token),
):
    # Get the session
    session = get_session_by_id(session_service, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    # Verify if the session's agent belongs to the user's client
    agent_id = uuid.UUID(session.app_name) if session.app_name else None
    if agent_id:
        agent = agent_service.get_agent(db, agent_id)
        if agent:
            await verify_user_client(payload, db, agent.client_id)

    return delete_session(session_service, session_id)
