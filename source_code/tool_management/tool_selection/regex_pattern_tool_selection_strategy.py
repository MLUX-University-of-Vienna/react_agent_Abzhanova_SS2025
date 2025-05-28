import re
from llm.llm_interface import LLMInterface
from typing import List, Optional
from tool_management.tools.agent_tool import AgentTool
from .tool_selection_strategy import ToolSelectionStrategy

class RegexPatternToolSelector(ToolSelectionStrategy):
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def gather_candidates(self, query: str, tools: dict[str, AgentTool]) -> List[str]:
        candidates = []
        for name, tool in tools.items():
            try:
                patterns = tool.get_patterns()
                for pattern in patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        candidates.append(name)
                        break
            except Exception as e:
                continue            
        return candidates

    def choose_tool(self, query: str, candidates: List[str], tools: dict[str, AgentTool]) -> Optional[str]:
        if candidates:
            if len(candidates) == 1:
                return candidates[0]
            return self._ask_llm_to_choose(query, candidates, tools)
        
        all_tool_names = list(tools.keys())
        return self._ask_llm_to_choose(query, all_tool_names, tools)
        
        
    def _ask_llm_to_choose(self, query: str, candidates: List[str], tools: dict[str, AgentTool]) -> Optional[str]:
        tool_descriptions = "\n".join(
            f"- {name}: {tools[name].description.strip()}" for name in candidates
        )

        prompt = f"""You are an intelligent agent assistant selecting the most suitable tool to use for a query.

Query: "{query}"

Candidate tools:
{tool_descriptions}

Which tool is best suited for this query? Respond with ONLY the tool name."""
        
        choice = self.llm.generate(prompt).strip()
        return choice if choice in candidates else None
        
    def ask_clarification(self, query: str) -> str:
        return f"I couldn't match your question \"{query}\" to any available tools. Could you clarify what you're asking?"