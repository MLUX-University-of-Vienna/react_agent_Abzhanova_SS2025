import re
from llm.llm_interface import LLMInterface
from typing import List, Optional
from tool_management.tools.agent_tool import AgentTool
from .tool_selection_strategy import ToolSelectionStrategy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

class RegexPatternToolSelector(ToolSelectionStrategy):
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def gather_candidates(self, query: str, tools: dict[str, AgentTool]) -> List[str]:
        candidates = []
        logger.debug(f"Gathering candidates for query: '{query}'")
        for name, tool in tools.items():
            try:
                patterns = tool.get_patterns()
                for pattern in patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        candidates.append(name)
                        logger.debug(f"Tool '{name}' matched pattern: '{pattern}'")
                        break
            except Exception as e:
                logger.warning(f"Failed to get patterns for tool '{name}': {e}")
                continue     
        logger.debug(f"Regex-matched candidates: {candidates}")       
        return candidates

    def choose_tool(self, query: str, candidates: List[str], tools: dict[str, AgentTool]) -> Optional[str]:
        if candidates:
            if len(candidates) == 1:
                logger.info(f"Only one candidate matched: {candidates[0]}")
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
        
        logger.info(f"LLM selected tool: '{choice}'")
        choice = self.llm.generate(prompt).strip()
        return choice if choice in candidates else None
        
    def ask_clarification(self, query: str) -> str:
        message = f"I couldn't match your question \"{query}\" to any available tools. Could you clarify what you're asking?"
        logger.info(f"Asking clarification: {message}")
        return message