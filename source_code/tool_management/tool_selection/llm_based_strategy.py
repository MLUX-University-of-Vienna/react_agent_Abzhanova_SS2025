import json
import logging
import re
from typing import List, Optional, Dict

from llm_adapters.llm_base import LLMBase
from tool_management.tools.agent_tool import AgentTool
from tool_management.tool_selection.tool_selection_strategy import ToolSelectionStrategy

class LLMToolSelectionStrategy(ToolSelectionStrategy):
    def __init__(self, llm: LLMBase):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self._last_clarification: Optional[str] = None

    def gather_candidates(self, query: str, tools: Dict[str, AgentTool]) -> List[str]:
        """
        Step 1: If the user said 'current', prioritize web_search.
        Otherwise regex‐match up to two tools.
        """
        q = query.lower()
        if "current " in q or "today" in q:
            # force live data for anything referencing 'current'
            return ["web_search"]

        matches = [
            name for name, tool in tools.items()
            if any(re.search(p, query, re.IGNORECASE) for p in tool.get_patterns())
        ]
        return matches[:2]

    def _llm_suggest_two(self, query: str, tools: Dict[str, AgentTool]) -> List[str]:
        """
        Step A: Ask the LLM to nominate *two* tools from the full set.
        """
        menu = []
        for name, t in tools.items():
            menu.append(
                f"{name}:\n"
                f"  desc: {t.description.strip()}\n"
                f"  example:\n{t.get_examples()}"
            )

        prompt = f"""SYSTEM: You are the router.  
Here are all available tools (name, description, one example):

{chr(10).join(menu)}

User’s query: "{query}"

STEP A: In a <think>…</think> block, pick the two tools you think are best suited,
and explain your reasoning.  
Then output exactly this JSON (no extra text):

```json
{{ "llm_candidates": ["<tool1>", "<tool2>"] }}
```"""

        resp = self.llm.generate(prompt)
        m = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
        if m:
            self.logger.info("Router thought A:\n" + m.group(1).strip())

        try:
            block = resp[resp.find("{"):resp.rfind("}")+1]
            data = json.loads(block)
            cands = data.get("llm_candidates", [])
            return [c for c in cands if c in tools][:2]
        except Exception as e:
            self.logger.warning("Could not parse llm_candidates: %s", e)
            return []

    def choose_tool(
        self,
        query: str,
        regex_candidates: List[str],
        tools: Dict[str, AgentTool]
    ) -> Optional[str]:
        """
        Step 2+3: Merge regex + LLM nominations (up to 4), then ask LLM to pick one.
        """
        llm_two = self._llm_suggest_two(query, tools)

        # merge regex first, then LLM picks, de‐dup
        merged = []
        for lst in (regex_candidates, llm_two):
            for name in lst:
                if name not in merged:
                    merged.append(name)
        merged = merged[:4]

        # If only one remains, pick it immediately
        if len(merged) == 1:
            return merged[0]

        # Build the final menu
        menu = []
        for name in merged:
            t = tools[name]
            menu.append(
                f"{name}:\n"
                f"  desc: {t.description.strip()}\n"
                f"  example:\n{t.get_examples()}"
            )

        prompt = f"""SYSTEM: You are the routing brain.  
Below are candidate tools (name, description, example):

{chr(10).join(menu)}

User’s query: "{query}"

STEP FINAL: In a <think>…</think> block, reason about which *one* tool is best,
then output exactly this JSON (no extra text):

```json
{{  
  "chosen_tool": "<tool_name or null>",  
  "clarification": "<a single question or null>"  
}}
```"""

        resp = self.llm.generate(prompt)
        m = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
        thought = m.group(1).strip().lower() if m else ""
        if m:
            self.logger.info("Router thought FINAL:\n" + m.group(1).strip())

        # ---- New: detect uncertainty in the chain-of-thought ----
        if any(keyword in thought for keyword in ("not sure", "uncertain", "maybe", "if")):
            self.logger.info("Router thought expresses uncertainty, asking user.")
            return None

        try:
            block = resp[resp.find("{"):resp.rfind("}")+1]
            data = json.loads(block)
        except Exception as e:
            self.logger.warning("Failed to parse final JSON: %s", e)
            data = {"chosen_tool": None, "clarification": None}

        chosen = data.get("chosen_tool")
        self._last_clarification = data.get("clarification")
        return chosen if chosen in merged else None

    def ask_clarification(self, query: str) -> str:
        """
        Step 4: Return the LLM's suggested clarification or a default prompt.
        """
        return self._last_clarification or (
            "I’m not entirely sure which tool to use. "
            "Could you clarify whether you’d like me to calculate something, "
            "retrieve live data, or look up a fact?"
        )
