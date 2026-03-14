"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ @author: Davidson Gomes                                                      │
│ @file: custom_tool.py                                                        │
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

from typing import Any, Callable, Dict, List
import requests
import json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AG2CustomToolBuilder:
    """Builds HTTP tools that can be registered with AG2 ConversableAgent instances."""

    def __init__(self):
        self.tools: List[Callable] = []

    def _create_http_tool(self, tool_config: Dict[str, Any]) -> Callable:
        """Create a plain Python callable suitable for AG2 tool registration."""
        name = tool_config["name"]
        description = tool_config["description"]
        endpoint = tool_config["endpoint"]
        method = tool_config["method"]
        headers = tool_config.get("headers", {})
        parameters = tool_config.get("parameters", {}) or {}
        values = tool_config.get("values", {})
        error_handling = tool_config.get("error_handling", {})

        path_params = parameters.get("path_params") or {}
        query_params = parameters.get("query_params") or {}
        body_params = parameters.get("body_params") or {}

        def http_tool(**kwargs) -> str:
            """Execute the HTTP request and return the result as a JSON string."""
            try:
                all_values = {**values, **kwargs}

                processed_headers = {
                    k: v.format(**all_values) if isinstance(v, str) else v
                    for k, v in headers.items()
                }

                url = endpoint
                for param in path_params:
                    if param in all_values:
                        url = url.replace(f"{{{param}}}", str(all_values[param]))

                query_params_dict: Dict[str, Any] = {}
                for param, value in query_params.items():
                    if isinstance(value, list):
                        query_params_dict[param] = ",".join(value)
                    elif param in all_values:
                        query_params_dict[param] = all_values[param]
                    else:
                        query_params_dict[param] = value

                for param, value in values.items():
                    if param not in query_params_dict and param not in path_params:
                        query_params_dict[param] = value

                body_data: Dict[str, Any] = {}
                for param in body_params:
                    if param in all_values:
                        body_data[param] = all_values[param]

                for param, value in values.items():
                    if (
                        param not in body_data
                        and param not in query_params_dict
                        and param not in path_params
                    ):
                        body_data[param] = value

                response = requests.request(
                    method=method,
                    url=url,
                    headers=processed_headers,
                    params=query_params_dict,
                    json=body_data or None,
                    timeout=error_handling.get("timeout", 30),
                )

                if response.status_code >= 400:
                    raise requests.exceptions.HTTPError(
                        f"Error in the request: {response.status_code} - {response.text}"
                    )

                try:
                    response_data = response.json()
                except ValueError:
                    response_data = {
                        "status_code": response.status_code,
                        "raw_response": response.text,
                    }
                return json.dumps(response_data)

            except Exception as e:
                logger.error(f"Error executing tool {name}: {str(e)}")
                return json.dumps(
                    error_handling.get(
                        "fallback_response",
                        {"error": "tool_execution_error", "message": str(e)},
                    )
                )

        http_tool.__name__ = name.replace(" ", "_")
        http_tool.__doc__ = description
        return http_tool

    def build_tools(self, tools_config: Dict[str, Any]) -> List[Callable]:
        """Build a list of callable tools from the agent config."""
        self.tools = []

        http_tools: List[Dict[str, Any]] = []
        if tools_config.get("http_tools"):
            http_tools = tools_config["http_tools"]
        elif tools_config.get("custom_tools") and tools_config["custom_tools"].get("http_tools"):
            http_tools = tools_config["custom_tools"]["http_tools"]
        elif (
            tools_config.get("tools")
            and isinstance(tools_config["tools"], dict)
            and tools_config["tools"].get("http_tools")
        ):
            http_tools = tools_config["tools"]["http_tools"]

        for http_tool_config in http_tools:
            self.tools.append(self._create_http_tool(http_tool_config))

        return self.tools
