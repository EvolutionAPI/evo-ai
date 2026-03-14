"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ @author: Davidson Gomes                                                      │
│ @file: service_providers.py                                                  │
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

import os
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.sessions import DatabaseSessionService
from google.adk.memory import InMemoryMemoryService
from dotenv import load_dotenv

load_dotenv()

from src.services.crewai.session_service import CrewSessionService
from src.services.ag2.session_service import AG2SessionService

if os.getenv("AI_ENGINE") == "crewai":
    session_service = CrewSessionService(db_url=os.getenv("POSTGRES_CONNECTION_STRING"))
elif os.getenv("AI_ENGINE") == "ag2":
    session_service = AG2SessionService(db_url=os.getenv("POSTGRES_CONNECTION_STRING"))
else:
    session_service = DatabaseSessionService(
        db_url=os.getenv("POSTGRES_CONNECTION_STRING")
    )

artifacts_service = InMemoryArtifactService()
memory_service = InMemoryMemoryService()
