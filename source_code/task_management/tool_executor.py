import re
import logging
from typing import Dict

from tool_management.tools.agent_tool import AgentTool, ToolResponse

class ToolExecutor:
    def __init__(self, tools: Dict[str, AgentTool]):
        self.tools = tools

    def execute(self, name: str, query: str) -> str:
        if name in self.tools:
            result = self.tools[name].execute(query)
            return result.content if result.success else f"Tool error: {result.content}"
        return "Invalid tool"