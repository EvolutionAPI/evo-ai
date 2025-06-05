"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ @author: Davidson Gomes                                                      │
│ @file: schemas.py                                                            │
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

from pydantic import BaseModel, Field, field_validator, model_validator, UUID4, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
import uuid
import re
from src.schemas.agent_config import LLMConfig, AgentConfig


class ClientBase(BaseModel):
    name: str
    email: Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if v is None:
            return v
        email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_regex, v):
            raise ValueError("Invalid email format")
        return v


class ClientCreate(ClientBase):
    pass


class Client(ClientBase):
    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ApiKeyBase(BaseModel):
    name: str
    provider: str


class ApiKeyCreate(ApiKeyBase):
    client_id: UUID4
    key_value: str


class ApiKeyUpdate(BaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    key_value: Optional[str] = None
    is_active: Optional[bool] = None


class ApiKey(ApiKeyBase):
    id: UUID4
    client_id: UUID4
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class AgentBase(BaseModel):
    name: Optional[str] = Field(
        None, description="Agent name (no spaces or special characters)"
    )
    description: Optional[str] = Field(None, description="Agent description")
    role: Optional[str] = Field(None, description="Agent role in the system")
    goal: Optional[str] = Field(None, description="Agent goal or objective")
    type: str = Field(
        ...,
        description="Agent type (llm, sequential, parallel, loop, a2a, workflow, task)",
    )
    model: Optional[str] = Field(
        None, description="LLM model identifier (required for LLM agents only)"
    )
    api_key_id: Optional[UUID4] = Field(
        None, description="Reference to a stored API Key ID"
    )
    instruction: Optional[str] = None
    agent_card_url: Optional[str] = Field(
        None, description="Agent card URL (required for a2a type)"
    )
    folder_id: Optional[UUID4] = Field(
        None, description="ID of the folder this agent belongs to"
    )
    config: Any = Field(None, description="Agent configuration based on type")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v, info):
        # Get values from validation context
        values = info.data if hasattr(info, 'data') else {}
        
        # A2A agents can have optional names
        if values.get("type") == "a2a":
            return v

        if not v:
            raise ValueError("Name is required for non-a2a agent types")

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Agent name cannot contain spaces or special characters")
        return v

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        valid_types = [
            "llm",
            "sequential", 
            "parallel", 
            "loop", 
            "a2a", 
            "workflow", 
            "task"
        ]
        if v not in valid_types:
            raise ValueError(
                f"Invalid agent type '{v}'. Must be one of: {', '.join(valid_types)}"
            )
        return v

    @field_validator("agent_card_url")
    @classmethod
    def validate_agent_card_url(cls, v, info):
        values = info.data if hasattr(info, 'data') else {}
        
        if values.get("type") == "a2a":
            if not v:
                raise ValueError("agent_card_url is required for a2a type agents")
            if not v.endswith("/.well-known/agent.json"):
                raise ValueError("agent_card_url must end with /.well-known/agent.json")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v, info):
        values = info.data if hasattr(info, 'data') else {}
        agent_type = values.get("type")
        
        if agent_type == "llm":
            # Para agentes LLM, o modelo é obrigatório e não pode ser vazio
            if not v or (isinstance(v, str) and v.strip() == ""):
                raise ValueError(
                    "LLM agents require a valid model configuration. "
                    "Please specify a model identifier (e.g., 'gpt-4', 'claude-3-sonnet', 'gemini-pro')"
                )
            
            # Verificar se o modelo tem um formato válido
            if isinstance(v, str) and len(v.strip()) < 3:
                raise ValueError("Model identifier must be at least 3 characters long")
        
        elif agent_type in ["workflow", "task", "sequential", "parallel", "loop"]:
            # Para estes tipos, não devem ter modelo
            if v and (isinstance(v, str) and v.strip()):
                # Avisar mas permitir (será removido durante a criação)
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"{agent_type} agents don't need model configuration. Model will be ignored.")
        
        return v

    @field_validator("api_key_id")
    @classmethod
    def validate_api_key_id(cls, v, info):
        values = info.data if hasattr(info, 'data') else {}
        agent_type = values.get("type")
        
        # API key é obrigatório para agentes LLM (a menos que esteja na config)
        if agent_type == "llm" and not v:
            # Verificar se tem API key na config
            config = values.get("config", {})
            if not config or not config.get("api_key"):
                # Não falhar aqui, deixar a validação para o momento da criação
                pass
        
        return v

    @field_validator("config")
    @classmethod
    def validate_config(cls, v, info):
        values = info.data if hasattr(info, 'data') else {}
        agent_type = values.get("type")
        
        if not agent_type:
            return v

        # A2A agents têm config opcional
        if agent_type == "a2a":
            return v or {}

        # Workflow agents têm config específico para workflow
        if agent_type == "workflow":
            if v and isinstance(v, dict):
                if not v.get("workflow"):
                    raise ValueError("Workflow agents must have 'workflow' configuration")
            return v

        # Config é obrigatório para outros tipos (exceto a2a)
        if not v and agent_type not in ["a2a"]:
            raise ValueError(
                f"Configuration is required for {agent_type} agent type"
            )

        # Validação específica por tipo
        if agent_type == "llm":
            return cls._validate_llm_config(v)
        elif agent_type in ["sequential", "parallel", "loop"]:
            return cls._validate_composite_config(v, agent_type)
        elif agent_type == "task":
            return cls._validate_task_config(v)

        return v

    @classmethod
    def _validate_llm_config(cls, v):
        """Valida configuração para agentes LLM"""
        if isinstance(v, dict):
            try:
                # Convert the dictionary to LLMConfig
                v = LLMConfig(**v)
            except Exception as e:
                raise ValueError(f"Invalid LLM configuration: {str(e)}")
        elif not isinstance(v, LLMConfig):
            raise ValueError("Invalid LLM configuration format")
        return v

    @classmethod
    def _validate_composite_config(cls, v, agent_type):
        """Valida configuração para agentes compostos (sequential, parallel, loop)"""
        if not isinstance(v, dict):
            raise ValueError(f'Configuration for {agent_type} agent must be a dictionary')
        
        if "sub_agents" not in v:
            raise ValueError(f'{agent_type} agents must have sub_agents configuration')
        
        if not isinstance(v["sub_agents"], list):
            raise ValueError("sub_agents must be a list")
        
        if not v["sub_agents"]:
            raise ValueError(
                f'{agent_type} agents must have at least one sub-agent'
            )
        
        # Validação específica para LoopAgent
        if agent_type == "loop":
            max_iterations = v.get("max_iterations", 5)
            if not isinstance(max_iterations, int) or max_iterations <= 0:
                raise ValueError("max_iterations must be a positive integer")
        
        return v

    @classmethod
    def _validate_task_config(cls, v):
        """Valida configuração para agentes de task"""
        if not isinstance(v, dict):
            raise ValueError('Configuration for task agent must be a dictionary')
        
        if "tasks" not in v:
            raise ValueError('Task agents must have tasks configuration')
        
        if not isinstance(v["tasks"], list):
            raise ValueError("tasks must be a list")
        
        if not v["tasks"]:
            raise ValueError('Task agents must have at least one task')
        
        # Validar cada task individualmente
        for i, task in enumerate(v["tasks"]):
            if not isinstance(task, dict):
                raise ValueError(f"Task {i+1} must be a dictionary")
            
            required_fields = ["agent_id", "description", "expected_output"]
            for field in required_fields:
                if field not in task:
                    raise ValueError(f"Task {i+1} missing required field: {field}")
                
                # Verificar se os campos não estão vazios
                if not task[field] or (isinstance(task[field], str) and not task[field].strip()):
                    raise ValueError(f"Task {i+1} field '{field}' cannot be empty")

        # Validar sub_agents se presente
        if "sub_agents" in v and v["sub_agents"] is not None:
            if not isinstance(v["sub_agents"], list):
                raise ValueError("sub_agents must be a list")

        return v

    @model_validator(mode='after')
    def validate_agent_consistency(self):
        """Validação cruzada entre campos do agente"""
        
        # Verificar consistência entre tipo e configurações
        if self.type == "llm":
            # LLM agents devem ter modelo
            if not self.model or (isinstance(self.model, str) and self.model.strip() == ""):
                raise ValueError("LLM agents must have a valid model")
            
            # LLM agents devem ter API key (na config ou api_key_id)
            has_api_key = bool(self.api_key_id)
            if not has_api_key and self.config:
                config_dict = self.config if isinstance(self.config, dict) else self.config.__dict__
                has_api_key = bool(config_dict.get("api_key"))
            
            if not has_api_key:
                raise ValueError("LLM agents must have an API key configured")
        
        elif self.type in ["workflow", "task", "sequential", "parallel", "loop"]:
            # Orchestrator agents não devem ter modelo
            if self.model and isinstance(self.model, str) and self.model.strip():
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"{self.type} agents don't need model configuration. Clearing model.")
                self.model = None
        
        elif self.type == "a2a":
            # A2A agents devem ter agent_card_url
            if not self.agent_card_url:
                raise ValueError("A2A agents must have agent_card_url")
        
        return self


class AgentCreate(AgentBase):
    client_id: UUID

    @model_validator(mode='after')
    def validate_creation_requirements(self):
        """Validações específicas para criação de agentes"""
        
        # Chamar validação da classe pai
        super().validate_agent_consistency()
        
        # Validações específicas para criação
        if self.type == "llm":
            # Para criação, ser mais rigoroso com modelo
            if not self.model or len(self.model.strip()) < 3:
                raise ValueError(
                    "LLM agents require a valid model identifier (minimum 3 characters). "
                    "Examples: 'gpt-4', 'claude-3-sonnet', 'gemini-pro'"
                )
        
        return self


class Agent(AgentBase):
    id: UUID
    client_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    agent_card_url: Optional[str] = None
    folder_id: Optional[UUID4] = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("agent_card_url", mode='before')
    @classmethod
    def set_agent_card_url(cls, v, info):
        if v:
            return v

        values = info.data if hasattr(info, 'data') else {}
        if "id" in values:
            from os import getenv
            return f"{getenv('API_URL', '')}/api/v1/a2a/{values['id']}/.well-known/agent.json"

        return v


class ToolConfig(BaseModel):
    id: str
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    inputModes: List[str] = Field(default_factory=list)
    outputModes: List[str] = Field(default_factory=list)


# Last edited by Arley Peter on 2025-05-17
class MCPServerBase(BaseModel):
    name: str
    description: Optional[str] = None
    config_type: str = Field(default="studio")
    config_json: Dict[str, Any] = Field(default_factory=dict)
    environments: Dict[str, Any] = Field(default_factory=dict)
    tools: Optional[List[ToolConfig]] = Field(default_factory=list) 
    type: str = Field(default="official")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("MCP Server name cannot be empty")
        return v.strip()

    @field_validator("config_type")
    @classmethod
    def validate_config_type(cls, v):
        valid_types = ["studio", "custom"]
        if v not in valid_types:
            raise ValueError(f"config_type must be one of: {valid_types}")
        return v

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        valid_types = ["official", "custom"]
        if v not in valid_types:
            raise ValueError(f"type must be one of: {valid_types}")
        return v


class MCPServerCreate(MCPServerBase):
    pass


class MCPServer(MCPServerBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ToolBase(BaseModel):
    name: str
    description: Optional[str] = None
    config_json: Dict[str, Any] = Field(default_factory=dict)
    environments: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()


class ToolCreate(ToolBase):
    pass


class Tool(ToolBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AgentFolderBase(BaseModel):
    name: str
    description: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Folder name cannot be empty")
        return v.strip()


class AgentFolderCreate(AgentFolderBase):
    client_id: UUID4


class AgentFolderUpdate(AgentFolderBase):
    pass


class AgentFolder(AgentFolderBase):
    id: UUID4
    client_id: UUID4
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AgentTypeInfo(BaseModel):
    """Informações sobre tipos de agente válidos"""
    type: str
    requires_model: bool
    requires_config: bool
    description: str

    @classmethod
    def get_valid_types(cls) -> Dict[str, 'AgentTypeInfo']:
        """Retorna informações sobre todos os tipos válidos de agente"""
        return {
            "llm": cls(
                type="llm",
                requires_model=True,
                requires_config=True,
                description="Large Language Model agent - requires model and API key"
            ),
            "workflow": cls(
                type="workflow",
                requires_model=False,
                requires_config=True,
                description="Workflow orchestrator agent - uses LangGraph for complex flows"
            ),
            "task": cls(
                type="task",
                requires_model=False,
                requires_config=True,
                description="Task management agent - coordinates multiple tasks"
            ),
            "sequential": cls(
                type="sequential",
                requires_model=False,
                requires_config=True,
                description="Sequential execution agent - runs sub-agents in order"
            ),
            "parallel": cls(
                type="parallel",
                requires_model=False,
                requires_config=True,
                description="Parallel execution agent - runs sub-agents concurrently"
            ),
            "loop": cls(
                type="loop",
                requires_model=False,
                requires_config=True,
                description="Loop execution agent - repeats sub-agents with conditions"
            ),
            "a2a": cls(
                type="a2a",
                requires_model=False,
                requires_config=False,
                description="Agent-to-Agent communication - external agent integration"
            )
        }


class ModelValidationResult(BaseModel):
    """Resultado da validação de modelo"""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    
    @classmethod
    def success(cls, warnings: List[str] = None) -> 'ModelValidationResult':
        return cls(is_valid=True, warnings=warnings or [])
    
    @classmethod
    def failure(cls, error_message: str) -> 'ModelValidationResult':
        return cls(is_valid=False, error_message=error_message)


class AgentValidationSummary(BaseModel):
    """Resumo de validação de agente"""
    agent_id: Optional[UUID] = None
    agent_name: Optional[str] = None
    agent_type: str
    is_valid: bool
    model_validation: ModelValidationResult
    config_validation: ModelValidationResult
    general_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)