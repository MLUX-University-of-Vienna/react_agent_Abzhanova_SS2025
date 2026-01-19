import logging
from datetime import datetime
from typing import Dict

from tool_management.tools.agent_tool import AgentTool
from agent_memory.data_structures import Evidence

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, tools: Dict[str, AgentTool]):
        self.tools = tools

    def execute(self, name: str, query: str) -> Evidence:
        tool = self.tools.get(name)
        if not tool:
            return Evidence(
                tool=name,
                content="Invalid tool",
                url=None,
                extracted={"error": "invalid_tool", "tool": name, "query": query},
                as_of=datetime.utcnow(),
                confidence=0.0,
            )

        try:
            tool_execution_evidence = tool.execute(query)

            if not isinstance(tool_execution_evidence, Evidence):
                logger.warning("Tool %s returned non-Evidence; normalizing", name)
                return Evidence(
                    tool=name,
                    content=str(tool_execution_evidence),
                    url=None,
                    extracted={"tool": name, "normalized": True, "query": query},
                    as_of=datetime.utcnow(),
                    confidence=0.6,
                )

            if not getattr(tool_execution_evidence, "tool", None):
                tool_execution_evidence = Evidence(
                    tool=name,
                    content=tool_execution_evidence.content,
                    url=tool_execution_evidence.url,
                    extracted=dict(tool_execution_evidence.extracted or {}),
                    as_of=getattr(tool_execution_evidence, "as_of", datetime.utcnow()),
                    confidence=getattr(tool_execution_evidence, "confidence", 0.7),
                )

            return tool_execution_evidence

        except Exception as e:
            logger.exception("Tool %s crashed", name)
            return Evidence(
                tool=name,
                content=f"Tool error: {e}",
                url=None,
                extracted={"error": "exception", "tool": name, "exception": repr(e), "query": query},
                as_of=datetime.utcnow(),
                confidence=0.0,
            )
