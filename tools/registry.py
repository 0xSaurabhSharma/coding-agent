#tools/registry.py
import logging
from typing import Any 
from pathlib import Path

from tools.base import Tool, ToolResult, ToolInvocation
from tools.builtin import ReadFileTool, get_all_builtin_tools


logger = logging.getLogger(__name__)

class ToolRegistry:

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning(f"Overwriting exsiting tool: {tool.name}")
        self._tools[tool.name] = tool 
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        if name in self._tools:
            del self._tools[name]
            return True 
        return False
    
    def get_all_tools(self) -> list[Tool]:
        tools: list[Tool] = []
        for tool in self._tools.values():
            tools.append(tool)
        return tools

    def get_schemas(self) -> list[dict[str,Any]]:
        return [ tool.to_openai_schema() for tool in self.get_all_tools()]

    def get_tool_by_name(self, name: str) -> Tool | None:
        if name in self._tools:
            return self._tools[name]
        return None

    async def invoke(
        self, 
        name: str, 
        params: dict[str,Any],
        cwd: Path, 
        # hook_system: HookSystem,
        # approval_manager: ApprovalManager | None = None
    ):
        tool = self.get_tool_by_name(name)
        if tool is None:
            return ToolResult.error_result(f"Unknown tool: {name}", metadata={"tool_name":name})
        
        # validate Tool 
        validation_errors = tool.validate_parameters(params)
        if validation_errors:
            result = ToolResult.error_result(
                f"Invalid parameters: {'; '.join(validation_errors)}",
                metadata={
                    "tool_name":name,
                    "validation_errors": validation_errors
                }
            )
            return result

        # execution 
        invocation = ToolInvocation(params, cwd)
        try:
            result = await tool.execute(invocation)
        except Exception as e:
            logger.exception(f"Tool {name} raised unexpected error")
            result = ToolResult.error_result(
                f"Internal Error: {str(e)}",
                metadata={
                    "tool_name": name
                }
            )
        return result


def create_default_registry()-> ToolRegistry:
    registery = ToolRegistry()

    for tool_class in get_all_builtin_tools():
        registery.register(tool_class())

    return registery