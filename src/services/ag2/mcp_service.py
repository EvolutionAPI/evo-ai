"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ @author: Davidson Gomes                                                      │
│ @file: mcp_service.py                                                        │
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

from typing import Any, Callable, Dict, List, Optional, Tuple
from src.utils.logger import setup_logger
from src.services.mcp_server_service import get_mcp_server
from sqlalchemy.orm import Session

logger = setup_logger(__name__)

try:
    from autogen import ConversableAgent
    from autogen.tools.experimental import McpServer

    HAS_AG2_MCP = True
except ImportError:
    logger.warning(
        "AG2 MCP support not available. Install ag2[mcp] to enable MCP tool connections."
    )
    HAS_AG2_MCP = False


class AG2MCPService:
    """
    Registers MCP server tools with AG2 ConversableAgent instances.

    AG2 exposes MCP tools via autogen.tools.experimental.McpServer (ag2[mcp] extra).
    Each connected server's tools are registered on the agent and a matching
    executor agent so that AG2 can invoke them during chat.
    """

    def __init__(self):
        self.tools: List[Any] = []

    async def build_tools(
        self,
        mcp_config: Dict[str, Any],
        db: Session,
    ) -> Tuple[List[Any], Optional[List[Any]]]:
        """
        Connect to configured MCP servers and collect tool objects.

        Returns (tools, server_list) where server_list holds open McpServer
        instances that the caller is responsible for closing.
        """
        if not HAS_AG2_MCP:
            logger.error("Cannot build AG2 MCP tools: ag2[mcp] is not installed")
            return [], None

        self.tools = []
        server_list: List[Any] = []

        mcp_servers = mcp_config.get("mcp_servers", [])
        if mcp_servers:
            for server_ref in mcp_servers:
                try:
                    mcp_server = get_mcp_server(db, server_ref["id"])
                    if not mcp_server:
                        logger.warning(f"MCP Server not found: {server_ref['id']}")
                        continue

                    server_config = mcp_server.config_json.copy()

                    # Resolve env@@ placeholders
                    if "env" in server_config and server_config["env"]:
                        for key, value in server_config["env"].items():
                            if value and value.startswith("env@@"):
                                env_key = value.replace("env@@", "")
                                if server_ref.get("envs") and env_key in server_ref.get("envs", {}):
                                    server_config["env"][key] = server_ref["envs"][env_key]
                                else:
                                    logger.warning(
                                        f"Environment variable '{env_key}' not provided for MCP server {mcp_server.name}"
                                    )

                    logger.info(f"Connecting to MCP server: {mcp_server.name}")
                    tools, server_instance = await self._connect_server(server_config)

                    if tools:
                        # Optionally filter to only the tools listed in the agent config
                        allowed = server_ref.get("tools", [])
                        if allowed:
                            tools = [t for t in tools if t.name in allowed]
                        self.tools.extend(tools)

                    if server_instance:
                        server_list.append(server_instance)
                        logger.info(
                            f"MCP server {mcp_server.name} connected. Added {len(tools)} tools."
                        )

                except Exception as e:
                    logger.error(
                        f"Error connecting to MCP server {server_ref.get('id', 'unknown')}: {e}"
                    )
                    continue

        custom_mcp_servers = mcp_config.get("custom_mcp_servers", [])
        if custom_mcp_servers:
            for server_conf in custom_mcp_servers:
                if not server_conf:
                    continue
                try:
                    tools, server_instance = await self._connect_server(server_conf)
                    if tools:
                        self.tools.extend(tools)
                    if server_instance:
                        server_list.append(server_instance)
                        logger.info(
                            f"Custom MCP server connected. Added {len(tools)} tools."
                        )
                except Exception as e:
                    logger.error(
                        f"Error connecting to custom MCP server {server_conf.get('url', 'unknown')}: {e}"
                    )
                    continue

        logger.info(f"AG2 MCP tools ready. Total: {len(self.tools)} tools.")
        return self.tools, server_list if server_list else None

    async def _connect_server(
        self, server_config: Dict[str, Any]
    ) -> Tuple[List[Any], Optional[Any]]:
        """Connect to a single MCP server and return its tools."""
        try:
            if "url" in server_config:
                server = McpServer({"url": server_config["url"]})
            else:
                command = server_config.get("command", "npx")
                args = server_config.get("args", [])
                env = server_config.get("env", {})
                server = McpServer({"command": command, "args": args, "env": env})

            tools = await server.list_tools()
            return tools, server

        except Exception as e:
            logger.error(f"Error connecting to MCP server: {e}")
            return [], None
