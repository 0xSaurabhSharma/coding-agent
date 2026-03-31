#tools/base.py
from __future__ import annotations
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, ValidationError
from pydantic.json_schema import model_json_schema, model_json_schema
# validation

class ToolKind(str, Enum):
    """
    """
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    NETWORK = "network"
    MEMORY = "memory"
    MCP = "mcp"


@dataclass
class ToolInvocation:
    params: dict[str, Any]
    cwd: Path


@dataclass
class ToolResult:
    # win, op, err, meta ?
    success: bool 
    output: str 
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False

    @classmethod
    def error_result(cls, error:str, output:str = "", **kwargs: Any):
        return cls(
            success=False,
            output=output,
            error=error,
            **kwargs,
        )
    
    @classmethod
    def success_result(cls, output:str, **kwargs: Any):
        return cls(
            success=True,
            output=output,
            error=None,
            **kwargs,
        )
    
    @classmethod
    def to_model_output(self) -> str:
        if self.success:
            return self.output
        return f"Error: {self.error}\n\nOutput:\n{self.output}"

    
@dataclass
class ToolConformation:
    tool_name: str 
    params: dict[str, Any]
    description: str


class Tool(ABC):
    name: str = "base_tool"
    description: str = "Base Tool"
    kind: ToolKind.READ
    
    def __init__(self) -> None:
        pass

    @property
    def schema(self) -> dict[str, Any] | type['BaseModel']:
        """
        Force tools to have schema.
            Dict: for MCPs
            BaseModels: for inbuilt tools
        """
        raise NotImplementedError("Tool must define schema property or class attributes")

    @abstractmethod
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        """thh"""
        pass

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """
            Validates llm's parameters before tool execution.                    
        """
        schema = self.schema 
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # pydantic validation 
            try:
                schema(**params)
            except ValidationError as e:
                errors = []
                for err in e.errors():
                    field = ".".join(str(x) for x in err.get('loc',[]))
                    msg = err.get("msg", "Validation error")
                    errors.append(f"Parameter '{field}': {msg}")
                
                return errors
            except Exception as e:
                return [str(e)]
        
        return []

    def is_mutating(self, params: dict[str, Any]) -> bool:
        return self.kind in { ToolKind.WRITE, ToolKind.NETWORK, ToolKind.SHELL, ToolKind.MEMORY }

    async def get_conformation(self,invocation: ToolInvocation) -> ToolInvocation | None:
        if not self.is_mutating(invocation.params):
            return None

        return ToolConformation(
            tool_name=self.name,
            params=invocation.params,
            description=f"Execute {self.name}"
        )
    
    def to_openai_schema(self) -> dict[str, Any]:
        schema = self.schema

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = model_json_schema(schema, mode="serialization")

            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": json_schema.get("properties", {}),
                    "required": json_schema.get("required", [])
                }
            }
        
        if isinstance(schema, dict):
            result = {
                "name": self.name,
                "description": self.description,
            }

            if "parameters" in schema:
                result["parameters"] = schema["parameters"]
            else:
                result["parameters"] = schema
            
            return result
        
        raise ValueError(f"Invalid schema type for tool {self.name}: {type(schema)}")