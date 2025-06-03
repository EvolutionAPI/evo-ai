"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ @author: Davidson Gomes                                                      │
│ @file: agent_builder.py                                                      │
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

from typing import List, Optional, Tuple
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents import SequentialAgent, ParallelAgent, LoopAgent, BaseAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.agent_tool import AgentTool
from src.schemas.schemas import Agent
from src.utils.logger import setup_logger
from src.core.exceptions import AgentNotFoundError
from src.services.agent_service import get_agent
from src.services.adk.custom_tools import CustomToolBuilder
from src.services.adk.mcp_service import MCPService
from src.services.adk.custom_agents.a2a_agent import A2ACustomAgent
from src.services.adk.custom_agents.workflow_agent import WorkflowAgent
from src.services.adk.custom_agents.task_agent import TaskAgent
from src.services.apikey_service import get_decrypted_api_key
from sqlalchemy.orm import Session
from contextlib import AsyncExitStack
from google.adk.tools import load_memory

from datetime import datetime
import uuid

from src.schemas.agent_config import AgentTask

logger = setup_logger(__name__)


class AgentBuilder:
    def __init__(self, db: Session):
        self.db = db
        self.custom_tool_builder = CustomToolBuilder()
        self.mcp_service = MCPService()

    async def _agent_tools_builder(self, agent) -> List[AgentTool]:
        """Build the tools for an agent."""
        agent_tools_ids = agent.config.get("agent_tools")
        agent_tools = []
        if agent_tools_ids and isinstance(agent_tools_ids, list):
            for agent_tool_id in agent_tools_ids:
                sub_agent = get_agent(self.db, agent_tool_id)
                llm_agent, _ = await self.build_llm_agent(sub_agent)
                if llm_agent:
                    agent_tools.append(AgentTool(agent=llm_agent))
        return agent_tools

    async def _create_llm_agent(
        self, agent, enabled_tools: List[str] = []
    ) -> Tuple[LlmAgent, Optional[AsyncExitStack]]:
        """Create an LLM agent from the agent data."""
        # Get custom tools from the configuration
        custom_tools = []
        custom_tools = self.custom_tool_builder.build_tools(agent.config)

        # Get MCP tools from the configuration
        mcp_tools = []
        mcp_exit_stack = None
        if agent.config.get("mcp_servers") or agent.config.get("custom_mcp_servers"):
            mcp_tools, mcp_exit_stack = await self.mcp_service.build_tools(
                agent.config, self.db
            )

        # Get agent tools
        agent_tools = await self._agent_tools_builder(agent)

        # Combine all tools
        all_tools = custom_tools + mcp_tools + agent_tools

        if enabled_tools:
            all_tools = [tool for tool in all_tools if tool.name in enabled_tools]
            logger.info(f"Enabled tools enabled. Total tools: {len(all_tools)}")

        now = datetime.now()
        current_datetime = now.strftime("%d/%m/%Y %H:%M")
        current_day_of_week = now.strftime("%A")
        current_date_iso = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M")

        # Substitute variables in the prompt
        formatted_prompt = agent.instruction.format(
            current_datetime=current_datetime,
            current_day_of_week=current_day_of_week,
            current_date_iso=current_date_iso,
            current_time=current_time,
        )

        # add role on beginning of the prompt
        if agent.role:
            formatted_prompt = (
                f"<agent_role>{agent.role}</agent_role>\n\n{formatted_prompt}"
            )

        # add goal on beginning of the prompt
        if agent.goal:
            formatted_prompt = (
                f"<agent_goal>{agent.goal}</agent_goal>\n\n{formatted_prompt}"
            )

        # Check if load_memory is enabled
        if agent.config.get("load_memory"):
            all_tools.append(load_memory)
            formatted_prompt = (
                formatted_prompt
                + "\n\n<memory_instructions>ALWAYS use the load_memory tool to retrieve knowledge for your context</memory_instructions>\n\n"
            )

        # Get API key from api_key_id
        api_key = None

        # Get API key from api_key_id
        if hasattr(agent, "api_key_id") and agent.api_key_id:
            decrypted_key = get_decrypted_api_key(self.db, agent.api_key_id)
            if decrypted_key:
                logger.info(f"Using stored API key for agent {agent.name}")
                api_key = decrypted_key
            else:
                logger.error(f"Stored API key not found for agent {agent.name}")
                raise ValueError(
                    f"API key with ID {agent.api_key_id} not found or inactive"
                )
        else:
            # Check if there is an API key in the config (temporary field)
            config_api_key = agent.config.get("api_key") if agent.config else None
            if config_api_key:
                logger.info(f"Using config API key for agent {agent.name}")
                # Check if it is a UUID of a stored key
                try:
                    key_id = uuid.UUID(config_api_key)
                    decrypted_key = get_decrypted_api_key(self.db, key_id)
                    if decrypted_key:
                        logger.info("Config API key is a valid reference")
                        api_key = decrypted_key
                    else:
                        # Use the key directly
                        api_key = config_api_key
                except (ValueError, TypeError):
                    # It is not a UUID, use directly
                    api_key = config_api_key
            else:
                logger.error(f"No API key configured for agent {agent.name}")
                raise ValueError(
                    f"Agent {agent.name} does not have a configured API key"
                )

        return (
            LlmAgent(
                name=agent.name,
                model=LiteLlm(model=agent.model, api_key=api_key),
                instruction=formatted_prompt,
                description=agent.description,
                tools=all_tools,
            ),
            mcp_exit_stack,
        )

    async def _get_sub_agents(
        self, sub_agent_ids: List[str]
    ) -> List[Tuple[LlmAgent, Optional[AsyncExitStack]]]:
        """Get and create LLM sub-agents."""
        sub_agents = []
        for sub_agent_id in sub_agent_ids:
            sub_agent_id_str = str(sub_agent_id)

            agent = get_agent(self.db, sub_agent_id_str)

            if agent is None:
                logger.error(f"Sub-agent not found: {sub_agent_id_str}")
                raise AgentNotFoundError(f"Agent with ID {sub_agent_id_str} not found")

            logger.info(f"Sub-agent found: {agent.name} (type: {agent.type})")

            if agent.type == "llm":
                sub_agent, exit_stack = await self._create_llm_agent(agent)
            elif agent.type == "a2a":
                sub_agent, exit_stack = await self.build_a2a_agent(agent)
            elif agent.type == "workflow":
                sub_agent, exit_stack = await self.build_workflow_agent(agent)
            elif agent.type == "task":
                sub_agent, exit_stack = await self.build_task_agent(agent)
            elif agent.type == "sequential":
                sub_agent, exit_stack = await self.build_composite_agent(agent)
            elif agent.type == "parallel":
                sub_agent, exit_stack = await self.build_composite_agent(agent)
            elif agent.type == "loop":
                sub_agent, exit_stack = await self.build_composite_agent(agent)
            else:
                raise ValueError(f"Invalid agent type: {agent.type}")

            sub_agents.append((sub_agent, exit_stack))
            logger.info(f"Sub-agent added: {agent.name}")

        logger.info(f"Sub-agents created: {len(sub_agents)}")
        logger.info(f"Sub-agents: {str(sub_agents)}")

        return sub_agents

    async def build_llm_agent(
        self, root_agent, enabled_tools: List[str] = []
    ) -> Tuple[LlmAgent, Optional[AsyncExitStack]]:
        """Build an LLM agent with its sub-agents."""
        logger.info("Creating LLM agent")

        sub_agents = []
        if root_agent.config.get("sub_agents"):
            sub_agents_with_stacks = await self._get_sub_agents(
                root_agent.config.get("sub_agents")
            )
            sub_agents = [agent for agent, _ in sub_agents_with_stacks]

        root_llm_agent, exit_stack = await self._create_llm_agent(
            root_agent, enabled_tools
        )
        if sub_agents:
            root_llm_agent.sub_agents = sub_agents

        return root_llm_agent, exit_stack

    async def build_a2a_agent(
        self, root_agent
    ) -> Tuple[BaseAgent, Optional[AsyncExitStack]]:
        """Build an A2A agent with its sub-agents."""
        logger.info(f"Creating A2A agent from {root_agent.agent_card_url}")

        if not root_agent.agent_card_url:
            raise ValueError("agent_card_url is required for a2a agents")

        try:
            sub_agents = []
            if root_agent.config.get("sub_agents"):
                sub_agents_with_stacks = await self._get_sub_agents(
                    root_agent.config.get("sub_agents")
                )
                sub_agents = [agent for agent, _ in sub_agents_with_stacks]

            config = root_agent.config or {}
            timeout = config.get("timeout", 300)

            a2a_agent = A2ACustomAgent(
                name=root_agent.name,
                agent_card_url=root_agent.agent_card_url,
                timeout=timeout,
                description=root_agent.description
                or f"A2A Agent for {root_agent.name}",
                sub_agents=sub_agents,
            )

            logger.info(
                f"A2A agent created successfully: {root_agent.name} ({root_agent.agent_card_url})"
            )

            return a2a_agent, None

        except Exception as e:
            logger.error(f"Error building A2A agent: {str(e)}")
            raise ValueError(f"Error building A2A agent: {str(e)}")

    async def build_workflow_agent(
        self, root_agent
    ) -> Tuple[WorkflowAgent, Optional[AsyncExitStack]]:
        """Build a workflow agent with its sub-agents."""
        logger.info(f"Creating Workflow agent from {root_agent.name}")

        agent_config = root_agent.config or {}

        if not agent_config.get("workflow"):
            raise ValueError("workflow is required for workflow agents")

        try:
            sub_agents = []
            if root_agent.config.get("sub_agents"):
                sub_agents_with_stacks = await self._get_sub_agents(
                    root_agent.config.get("sub_agents")
                )
                sub_agents = [agent for agent, _ in sub_agents_with_stacks]

            config = root_agent.config or {}
            timeout = config.get("timeout", 300)

            workflow_agent = WorkflowAgent(
                name=root_agent.name,
                flow_json=agent_config.get("workflow"),
                timeout=timeout,
                description=root_agent.description
                or f"Workflow Agent for {root_agent.name}",
                sub_agents=sub_agents,
                db=self.db,
            )

            logger.info(f"Workflow agent created successfully: {root_agent.name}")

            return workflow_agent, None

        except Exception as e:
            logger.error(f"Error building Workflow agent: {str(e)}")
            raise ValueError(f"Error building Workflow agent: {str(e)}")

    async def build_task_agent(
        self, root_agent
    ) -> Tuple[TaskAgent, Optional[AsyncExitStack]]:
        """Build a task agent with its sub-agents."""
        logger.info(f"Creating Task agent: {root_agent.name}")

        agent_config = root_agent.config or {}

        if not agent_config.get("tasks"):
            raise ValueError("tasks are required for Task agents")

        try:
            # Get sub-agents if there are any
            sub_agents = []
            if root_agent.config.get("sub_agents"):
                sub_agents_with_stacks = await self._get_sub_agents(
                    root_agent.config.get("sub_agents")
                )
                sub_agents = [agent for agent, _ in sub_agents_with_stacks]

            # Additional configurations
            config = root_agent.config or {}

            # Convert tasks to the expected format by TaskAgent
            tasks = []
            for task_config in config.get("tasks", []):
                task = AgentTask(
                    agent_id=task_config.get("agent_id"),
                    description=task_config.get("description", ""),
                    expected_output=task_config.get("expected_output", ""),
                    enabled_tools=task_config.get("enabled_tools", []),
                )
                tasks.append(task)

            # Create the Task agent
            task_agent = TaskAgent(
                name=root_agent.name,
                tasks=tasks,
                db=self.db,
                sub_agents=sub_agents,
            )

            logger.info(f"Task agent created successfully: {root_agent.name}")

            return task_agent, None

        except Exception as e:
            logger.error(f"Error building Task agent: {str(e)}")
            raise ValueError(f"Error building Task agent: {str(e)}")

    async def build_composite_agent(
        self, root_agent
    ) -> Tuple[SequentialAgent | ParallelAgent | LoopAgent, Optional[AsyncExitStack]]:
        """Build a composite agent (Sequential, Parallel or Loop) with its sub-agents."""
        logger.info(
            f"Processing sub-agents for agent {root_agent.type} (ID: {root_agent.id}, Name: {root_agent.name})"
        )

        if not root_agent.config.get("sub_agents"):
            logger.error(
                f"Sub_agents configuration not found or empty for agent {root_agent.name}"
            )
            raise ValueError(f"Missing sub_agents configuration for {root_agent.name}")

        logger.info(
            f"Sub-agents IDs to be processed: {root_agent.config.get('sub_agents', [])}"
        )

        sub_agents_with_stacks = await self._get_sub_agents(
            root_agent.config.get("sub_agents", [])
        )

        logger.info(
            f"Sub-agents processed: {len(sub_agents_with_stacks)} of {len(root_agent.config.get('sub_agents', []))}"
        )

        sub_agents = [agent for agent, _ in sub_agents_with_stacks]
        logger.info(f"Extracted sub-agents: {[agent.name for agent in sub_agents]}")

        if root_agent.type == "sequential":
            logger.info(f"Creating SequentialAgent with {len(sub_agents)} sub-agents")
            return (
                SequentialAgent(
                    name=root_agent.name,
                    sub_agents=sub_agents,
                    description=root_agent.config.get("description", ""),
                ),
                None,
            )
        elif root_agent.type == "parallel":
            logger.info(f"Creating ParallelAgent with {len(sub_agents)} sub-agents")
            return (
                ParallelAgent(
                    name=root_agent.name,
                    sub_agents=sub_agents,
                    description=root_agent.config.get("description", ""),
                ),
                None,
            )
        elif root_agent.type == "loop":
            logger.info(f"Creating LoopAgent with {len(sub_agents)} sub-agents")
            return (
                LoopAgent(
                    name=root_agent.name,
                    sub_agents=sub_agents,
                    description=root_agent.config.get("description", ""),
                    max_iterations=root_agent.config.get("max_iterations", 5),
                ),
                None,
            )
        else:
            raise ValueError(f"Invalid agent type: {root_agent.type}")

    async def build_agent(self, root_agent, enabled_tools: List[str] = []) -> Tuple[
        LlmAgent
        | SequentialAgent
        | ParallelAgent
        | LoopAgent
        | A2ACustomAgent
        | WorkflowAgent
        | TaskAgent,
        Optional[AsyncExitStack],
    ]:
        """Build the appropriate agent based on the type of the root agent."""
        if root_agent.type == "llm":
            return await self.build_llm_agent(root_agent, enabled_tools)
        elif root_agent.type == "a2a":
            return await self.build_a2a_agent(root_agent)
        elif root_agent.type == "workflow":
            return await self.build_workflow_agent(root_agent)
        elif root_agent.type == "task":
            return await self.build_task_agent(root_agent)
        else:
            return await self.build_composite_agent(root_agent)
