import json
import logging
from typing import List, Dict
from llm.llm_interface import LLMInterface
from .query_separation_strategy import QuerySeparationStrategy

class LLMQuerySeparationStrategy(QuerySeparationStrategy):
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def separate(self, query: str) -> List[str]:
        prompt = f"""SYSTEM: You are a question decomposer.
Split the user's query into independent sub-questions, each of which can be answered on its own.

**Output exactly this JSON and nothing else** (no commentary, no <think> blocks):

{{ "subqueries": ["<first subquery>", "<second subquery>", ...] }}

Query: "{query}"
"""
        resp = self.llm.generate(prompt)

        # find the first JSON object by manual brace counting
        start = resp.find("{")
        if start != -1:
            depth = 0
            for i, ch in enumerate(resp[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        json_block = resp[start : i + 1]
                        try:
                            data: Dict = json.loads(json_block)
                            subs = data.get("subqueries", [])
                            if isinstance(subs, list) and subs:
                                return subs
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse subqueries JSON: {e}")
                        break

        # fallback: treat whole query as a single subquery
        self.logger.warning("Could not extract subqueries, falling back to the full query.")
        return [query]