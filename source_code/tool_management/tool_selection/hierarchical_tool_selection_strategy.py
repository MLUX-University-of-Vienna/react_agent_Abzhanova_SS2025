import json
import re
import logging
from typing import List, Optional, Dict
from llm.llm_interface import LLMInterface
from tool_management.tools.agent_tool import AgentTool
from tool_management.tool_selection.tool_selection_strategy import ToolSelectionStrategy

class HierarchicalToolSelectionStrategy(ToolSelectionStrategy):
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self._last_clarification: Optional[str] = None
        self._classification_cache: Dict[str, str] = {}

    def gather_candidates(self, query: str, tools: Dict[str, AgentTool]) -> List[str]:
        """Gather tools whose patterns match the query, with fallbacks."""
        self.logger.info(f"Gathering candidates for query: '{query}'")
        
        candidates = []
        for tool_name, tool in tools.items():
            patterns = tool.get_patterns()
            if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in patterns):
                candidates.append(tool_name)
                self.logger.debug(f"Tool '{tool_name}' matched query patterns")
        
        # Always include essential tools as fallbacks
        essential_tools = ['web_search', 'calculator']
        for tool_name in essential_tools:
            if tool_name in tools and tool_name not in candidates:
                candidates.append(tool_name)
                self.logger.debug(f"Added essential tool '{tool_name}' as fallback")
        
        self.logger.info(f"Final candidates: {candidates}")
        return candidates

    def choose_tool(
        self,
        query: str,
        candidates: List[str],
        tools: Dict[str, AgentTool]
    ) -> Optional[str]:
        """Improved tool selection with better classification and prioritization."""
        # Stage 1: Classification with caching
        if query in self._classification_cache:
            category = self._classification_cache[query]
            self.logger.info(f"Using cached classification: {category}")
        else:
            category = self._classify_query(query)
            self._classification_cache[query] = category
        
        # Stage 2: Tool selection based on category
        if category == "computation":
            if "calculator" in candidates:
                self.logger.info("Selecting calculator for computation task")
                return "calculator"
        
        elif category == "retrieval":
            # Prioritize tools based on query content
            query_lower = query.lower()
            if "current" in query_lower or "latest" in query_lower:
                if "web_search" in candidates:
                    self.logger.info("Prioritizing web_search for current information")
                    return "web_search"
            
            if "who is" in query_lower or "what is" in query_lower:
                if "wikipedia" in candidates:
                    self.logger.info("Prioritizing wikipedia for factual lookup")
                    return "wikipedia"
            
            # Fallback to LLM selection if no clear preference
            return self._select_with_llm(query, candidates, tools)
        
        self.logger.info("No specialized tool needed for general query")
        return None

    def _classify_query(self, query: str) -> str:
        """Classify query intent with more precise prompting."""
        prompt = f"""Classify this query's intent:
1. "computation" - requires math/calculation
2. "retrieval" - requires looking up facts/current info
3. "general" - can be answered directly

Respond ONLY with JSON: {{"category": "<computation|retrieval|general>", "reason": "<brief explanation>"}}

Query: {query}"""
        
        try:
            response = self.llm.generate(prompt)
            data = self._parse_json(response)
            category = data.get("category", "").lower()
            self.logger.debug(f"Classification reason: {data.get('reason', '')}")
            return category if category in ["computation", "retrieval", "general"] else "general"
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return "general"

    def _select_with_llm(self, query: str, candidates: List[str], tools: Dict[str, AgentTool]) -> Optional[str]:
        """Use LLM to select between multiple retrieval tools."""
        tool_menu = "\n".join(
            f"- {name}: {tools[name].description}" 
            for name in candidates 
            if name in tools
        )
        
        prompt = f"""Select the SINGLE BEST tool for this query from:
{tool_menu}

Consider:
1. For current/live info: prefer web_search
2. For established facts: prefer wikipedia
3. For domain-specific docs: prefer vector_search

Respond ONLY with JSON: {{"tool": "<tool_name>", "reason": "<brief justification>"}}

Query: {query}"""
        
        try:
            response = self.llm.generate(prompt)
            data = self._parse_json(response)
            selected = data.get("tool", "")
            self.logger.debug(f"Selection reason: {data.get('reason', '')}")
            return selected if selected in candidates else None
        except Exception as e:
            self.logger.error(f"Tool selection failed: {e}")
            return None

    def ask_clarification(self, query: str) -> str:
        """Generate more specific clarification prompts."""
        query_lower = query.lower()
        if "current" in query_lower or "latest" in query_lower:
            self._last_clarification = "Are you looking for the most up-to-date information on this topic?"
        elif "who is" in query_lower:
            self._last_clarification = "Would you like biographical information about this person?"
        else:
            self._last_clarification = "Should I look up current information, historical facts, or perform a calculation?"
        
        return self._last_clarification

    def _parse_json(self, text: str) -> Dict[str, str]:
        """Robust JSON parsing from LLM responses."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start < 0 or end <= 0:
                raise ValueError("No JSON found")
            return json.loads(text[start:end])
        except Exception as e:
            self.logger.error(f"JSON parse error: {e}")
            return {}