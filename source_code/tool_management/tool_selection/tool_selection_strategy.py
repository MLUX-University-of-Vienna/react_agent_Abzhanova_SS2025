from abc import ABC, abstractmethod
from typing import List, Optional
from ..tools.agent_tool import AgentTool
from llm.llm_interface import LLMInterface

class ToolSelectionStrategy(ABC):
    @abstractmethod
    def gather_candidates(self, query: str, tools: dict[str, AgentTool]) -> List[str]:
        """Return a list of candidate tool names whose patterns match the query."""
        pass

    @abstractmethod
    def choose_tool(
        self, 
        query: str, 
        candidates: List[str], 
        tools: dict[str, AgentTool]
    ) -> Optional[str]:
        """Given multiple candidates, pick one or return None."""
        pass

    @abstractmethod
    def ask_clarification(self, query: str) -> str:
        """Return a clarification prompt for the user."""
        pass
