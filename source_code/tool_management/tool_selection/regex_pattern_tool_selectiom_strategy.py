import re
from llm.llm_interface import LLMInterface
from typing import List

class RegexPatternToolSelector(ToolSelectionStrategy):
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def gather_candidates(self, query: str, tools: dict[str, AgentTool]) -> List[str]:
        candidates = []
        for name, tool in tools.items():
            patterns = tool.get_patterns()
                for pattern in patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        candidates.append(name)
        return candidates