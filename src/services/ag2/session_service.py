"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ @author: Davidson Gomes                                                      │
│ @file: session_service.py                                                    │
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
│ distributed under the License is distributed on an "AS IS" BASIS,           │
│ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    │
│ See the License for the specific language governing permissions and          │
│ limitations under the License.                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ @important                                                                   │
│ For any future changes to the code in this file, it is recommended to        │
│ include, together with the modification, the information of the developer    │
│ who changed it and the date of modification.                                 │
└──────────────────────────────────────────────────────────────────────────────┘
"""

from datetime import datetime
import json
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, Text
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import (
    sessionmaker,
    DeclarativeBase,
    Mapped,
    mapped_column,
)
from sqlalchemy.sql import func
from sqlalchemy.types import DateTime, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import TypeDecorator

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DynamicJSON(TypeDecorator):
    """JSON type that uses JSONB in PostgreSQL and TEXT with JSON serialization elsewhere."""

    impl = Text

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(postgresql.JSONB)
        else:
            return dialect.type_descriptor(Text)

    def process_bind_param(self, value, dialect):
        if value is not None:
            if dialect.name == "postgresql":
                return value
            else:
                return json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            if dialect.name == "postgresql":
                return value
            else:
                return json.loads(value)
        return value


class Base(DeclarativeBase):
    pass


class AG2StorageSession(Base):
    """Stores AG2 conversation sessions in PostgreSQL."""

    __tablename__ = "ag2_sessions"

    app_name: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    messages: Mapped[MutableDict[str, Any]] = mapped_column(
        MutableDict.as_mutable(DynamicJSON), default=[]
    )
    create_time: Mapped[DateTime] = mapped_column(DateTime(), default=func.now())
    update_time: Mapped[DateTime] = mapped_column(
        DateTime(), default=func.now(), onupdate=func.now()
    )

    def __repr__(self):
        return f"<AG2StorageSession(id={self.id})>"


class AG2Session:
    """In-memory representation of an AG2 session."""

    def __init__(self, app_name: str, user_id: str, session_id: str):
        self.app_name = app_name
        self.user_id = user_id
        self.id = session_id
        self.messages: List[Dict[str, Any]] = []


class AG2SessionService:
    """Session service for AG2 engine — stores conversation history in PostgreSQL."""

    def __init__(self, db_url: str):
        try:
            self.engine = create_engine(db_url)
        except Exception as e:
            raise ValueError(f"Failed to create database engine: {e}")

        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"AG2SessionService started with database at {db_url}")

    def get_or_create(self, agent_id: str, external_id: str) -> AG2Session:
        """Retrieve an existing session or create a new one."""
        session_id = f"{external_id}_{agent_id}"
        with self.SessionLocal() as db:
            record = db.get(AG2StorageSession, (agent_id, external_id, session_id))
            if record is None:
                record = AG2StorageSession(
                    app_name=agent_id,
                    user_id=external_id,
                    id=session_id,
                    messages=[],
                )
                db.add(record)
                db.commit()
                db.refresh(record)
                logger.info(
                    f"Created new AG2 session {session_id} for agent {agent_id} / user {external_id}"
                )

        session = AG2Session(app_name=agent_id, user_id=external_id, session_id=session_id)
        # Load persisted messages
        with self.SessionLocal() as db:
            record = db.get(AG2StorageSession, (agent_id, external_id, session_id))
            if record and record.messages:
                session.messages = list(record.messages) if isinstance(record.messages, list) else []
        return session

    def build_messages(self, session: AG2Session) -> List[Dict[str, Any]]:
        """
        Return the conversation history as a list of AG2-compatible message dicts.
        Each entry is {"role": "user"|"assistant", "content": <str>}.
        """
        return list(session.messages)

    def append(self, session: AG2Session, role: str, content: str) -> None:
        """Append a new message to the in-memory session."""
        session.messages.append({"role": role, "content": content})

    def save(self, session: AG2Session) -> None:
        """Persist the session messages to PostgreSQL."""
        with self.SessionLocal() as db:
            record = db.get(
                AG2StorageSession, (session.app_name, session.user_id, session.id)
            )
            if record is None:
                logger.error(f"AG2 session not found for save: {session.id}")
                return
            record.messages = list(session.messages)
            db.commit()
            db.refresh(record)
        logger.info(
            f"AG2 session {session.id} saved with {len(session.messages)} messages"
        )

    def delete(self, agent_id: str, external_id: str) -> bool:
        """Delete all sessions for a given agent + user pair."""
        from sqlalchemy import delete as sa_delete

        session_id = f"{external_id}_{agent_id}"
        with self.SessionLocal() as db:
            stmt = sa_delete(AG2StorageSession).where(
                AG2StorageSession.app_name == agent_id,
                AG2StorageSession.user_id == external_id,
                AG2StorageSession.id == session_id,
            )
            result = db.execute(stmt)
            db.commit()
            logger.info(f"AG2 session {session_id} deleted")
            return result.rowcount > 0
