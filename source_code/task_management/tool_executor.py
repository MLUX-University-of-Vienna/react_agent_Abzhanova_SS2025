import logging
from datetime import datetime

from tool_management.tools.agent_tool import AgentTool
from agent_core.models import ToolResponse, Observation

logger = logging.getLogger(__name__)

class ToolExecutor:
    def __init__(self, tools: dict[str, AgentTool]):
        self.tools = tools

    def execute(self, name: str, query: str) -> ToolResponse:
        tool = self.tools.get(name)
        if not tool:
            return ToolResponse(
                success=False,
                observations=[Observation(raw_text="Invalid tool", meta={"tool": name})],
                metadata={"tool": name}
            )

        try:
            resp = tool.execute(query)
            if isinstance(resp, ToolResponse):
                # ensure minimal provenance defaults if missing
                if not resp.metadata.get("tool"):
                    resp.metadata["tool"] = name
                return resp

            logger.warning("Tool %s returned non-ToolResponse; normalizing", name)
            return ToolResponse(
                success=True,
                observations=[Observation(
                    raw_text=str(resp),
                    source=name,
                    observed_at=datetime.utcnow()
                )],
                metadata={"tool": name, "normalized": True}
            )
        except Exception as e:
            logger.exception("Tool %s crashed", name)
            return ToolResponse(
                success=False,
                observations=[Observation(raw_text=f"Tool error: {e}")],
                metadata={"tool": name, "error": repr(e)}
            )
