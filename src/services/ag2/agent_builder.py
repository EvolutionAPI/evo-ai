import uuid
from typing import Tuple, Optional
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import DefaultPattern, AutoPattern
from autogen.agentchat.group import (
    ContextVariables,
    RevertToUserTarget,
    TerminateTarget,
    AgentTarget,
    OnCondition,
    StringLLMCondition,
    OnContextCondition,
    ExpressionContextCondition,
    ContextExpression,
)
from sqlalchemy.orm import Session
from src.services.agent_service import get_agent
from src.services.apikey_service import get_decrypted_api_key
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AG2AgentBuilder:
    def __init__(self, db: Session):
        self.db = db

    async def _get_api_key(self, agent) -> str:
        """Reuse the same key resolution logic as ADK and CrewAI builders."""
        if hasattr(agent, "api_key_id") and agent.api_key_id:
            key = get_decrypted_api_key(self.db, agent.api_key_id)
            if key:
                return key
            raise ValueError(f"API key {agent.api_key_id} not found or inactive")
        config_key = agent.config.get("api_key") if agent.config else None
        if config_key:
            try:
                key = get_decrypted_api_key(self.db, uuid.UUID(config_key))
                return key or config_key
            except (ValueError, TypeError):
                return config_key
        raise ValueError(f"No API key configured for agent {agent.name}")

    def _build_llm_config(self, agent, api_key: str) -> LLMConfig:
        return LLMConfig({"model": agent.model, "api_key": api_key})

    def _build_system_message(self, agent) -> str:
        parts = []
        if agent.role:
            parts.append(f"Role: {agent.role}")
        if agent.goal:
            parts.append(f"Goal: {agent.goal}")
        if agent.instruction:
            parts.append(agent.instruction)
        return "\n\n".join(parts)

    async def build_conversable_agent(self, agent) -> ConversableAgent:
        api_key = await self._get_api_key(agent)
        # AG2 0.11+ rejects names containing whitespace for OpenAI models
        safe_name = agent.name.replace(" ", "_")
        return ConversableAgent(
            name=safe_name,
            system_message=self._build_system_message(agent),
            description=agent.description or "",
            llm_config=self._build_llm_config(agent, api_key),
        )

    def _apply_handoffs(self, ca: ConversableAgent, config: dict, all_agents: dict):
        """
        Apply AG2 handoff conditions from the agent config's optional 'handoffs' field.

        Config format:
        {
          "handoffs": [
            {
              "type": "llm",
              "target_agent_id": "<uuid>",
              "condition": "Route when the user asks about billing"
            },
            {
              "type": "context",
              "target_agent_id": "<uuid>",
              "expression": "${is_vip} == True"
            }
          ],
          "after_work": "revert_to_user"  // or "terminate"
        }
        """
        handoffs_config = config.get("handoffs", [])
        llm_conditions = []
        context_conditions = []

        for h in handoffs_config:
            target_id = h.get("target_agent_id")
            target_agent = all_agents.get(str(target_id))
            if not target_agent:
                logger.warning(f"Handoff target {target_id} not found, skipping")
                continue

            if h["type"] == "llm":
                llm_conditions.append(
                    OnCondition(
                        target=AgentTarget(target_agent),
                        condition=StringLLMCondition(prompt=h["condition"]),
                    )
                )
            elif h["type"] == "context":
                context_conditions.append(
                    OnContextCondition(
                        target=AgentTarget(target_agent),
                        condition=ExpressionContextCondition(
                            expression=ContextExpression(h["expression"])
                        ),
                    )
                )

        if llm_conditions:
            ca.handoffs.add_llm_conditions(llm_conditions)
        if context_conditions:
            ca.handoffs.add_context_conditions(context_conditions)

        after_work = config.get("after_work", "revert_to_user")
        if after_work == "terminate":
            ca.handoffs.set_after_work(TerminateTarget())
        else:
            ca.handoffs.set_after_work(RevertToUserTarget())

    async def build_group_chat_setup(self, root_agent) -> dict:
        """
        Build a GroupChat pattern from an agent record with sub_agents.
        Returns a dict consumed by the runner's initiate_group_chat call.
        """
        config = root_agent.config or {}
        sub_agent_ids = config.get("sub_agents", [])
        if not sub_agent_ids:
            raise ValueError("group_chat agent requires at least one sub_agent")

        # Build all sub-agents first so handoff resolution can reference them
        all_agents = {}
        agents = []
        for aid in sub_agent_ids:
            db_agent = get_agent(self.db, str(aid))
            if db_agent is None:
                raise ValueError(f"Sub-agent {aid} not found")
            ca = await self.build_conversable_agent(db_agent)
            all_agents[str(aid)] = ca
            agents.append(ca)

        root_ca = await self.build_conversable_agent(root_agent)
        all_agents[str(root_agent.id)] = root_ca

        # Apply handoffs to each agent if configured
        for aid in sub_agent_ids:
            db_agent = get_agent(self.db, str(aid))
            if db_agent and db_agent.config:
                self._apply_handoffs(all_agents[str(aid)], db_agent.config, all_agents)

        api_key = await self._get_api_key(root_agent)
        manager_llm = self._build_llm_config(root_agent, api_key)

        pattern_type = config.get("pattern", "auto")
        if pattern_type == "auto":
            pattern = AutoPattern(
                initial_agent=root_ca,
                agents=[root_ca] + agents,
                group_manager_args={"llm_config": manager_llm},
            )
        else:
            pattern = DefaultPattern(
                initial_agent=root_ca,
                agents=[root_ca] + agents,
                group_after_work=RevertToUserTarget(),
            )

        return {
            "pattern": pattern,
            "agents": [root_ca] + agents,
            "max_rounds": config.get("max_rounds", 10),
            "context_variables": ContextVariables(
                data=config.get("context_variables", {})
            ),
        }

    async def build_agent(self, root_agent) -> Tuple[object, None]:
        """
        Entry point matching the ADK/CrewAI AgentBuilder interface.
        Returns (agent_or_setup_dict, exit_stack).

        Orchestration mode is read from config["ag2_mode"]:
          "group_chat" → GroupChat with sub-agents from config["sub_agents"]
          "single" / absent → single ConversableAgent (default)
        No new agent type is required in the DB; all AG2 agents use type="llm".
        """
        ag2_mode = (root_agent.config or {}).get("ag2_mode", "single")
        if ag2_mode == "group_chat":
            return await self.build_group_chat_setup(root_agent), None
        else:
            return await self.build_conversable_agent(root_agent), None
