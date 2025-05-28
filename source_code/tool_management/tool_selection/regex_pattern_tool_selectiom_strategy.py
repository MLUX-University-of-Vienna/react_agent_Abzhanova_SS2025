import re
from llm.llm_interface import LLMInterface
from typing import List, 

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
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        tool_descriptions = "\n".join(
            f"- {name}: {tools[name].description}" for name in candidates
        )

        prompt = f"""You are an intelligent agent assistant helping to select the most suitable tool for a task.

Query: {query}

Candidate tools:
{tool_descriptions}

Which tool is best suited for this query? Return only the tool name."""
        
        choice = self.llm.generate(prompt).strip()
        return choice if choice in candidates else None