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