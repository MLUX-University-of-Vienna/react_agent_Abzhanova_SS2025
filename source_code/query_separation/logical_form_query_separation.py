import json
import logging
import re
from typing import List, Dict

from llm.llm_interface import LLMInterface
from .query_separation_strategy import QuerySeparationStrategy

import json
import logging
from typing import List, Dict

from llm.llm_interface import LLMInterface
from query_separation.query_separation_strategy import QuerySeparationStrategy

class LLMLogicalSeparationStrategy(QuerySeparationStrategy):
    """
    Splits a complex query by having the LLM emit a minimal 'field(entity) ∧ ...' DSL,
    then converts each field-entity pair into a natural-language subtask.
    """
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def separate(self, query: str) -> List[str]:
        # 1) Prompting LLM to produce minimal DSL
        dsl_prompt = (
            "SYSTEM: You are a logic parser. Convert the user's question into a logical form using the following rules:\n"
            "- Use field(entity) or field(x) for variables.\n"
            "- Use logical operators: ∧ (and), ∨ (or), ¬ (not), ⇒ (implies), =, >, <.\n"
            "- Resolve pronouns (like 'he', 'they') to what they refer to.\n"
            "- Keep the form clean and symbolic. Return only the logic, not explanations.\n"
            f"User query: '{query}'\n"
            "Return logical form:"
        )


        self.logger.debug("Separation prompt:%s", dsl_prompt)
        logic = self.llm.generate(dsl_prompt).strip()
        self.logger.debug("Received logic DSL: %s", logic)

        # 2) Split into segments
        segments = [seg.strip() for seg in logic.split('∧') if seg.strip()]

        if not segments:
            self.logger.warning("No valid DSL segments found; falling back to punctuation-based splitting.")
            return [part.strip() for part in re.split(r'[,.]', query) if part.strip()]

        rephrase_prompt = (
            "SYSTEM: Convert each logical expression of the form field(entity) into a clear and minimal natural-language question.\n"
            "Each line must correspond to one field(entity) pair.\n"
            f"DSL: {logic}\n"
            "Questions:"
        )


        self.logger.debug("Rephrasing prompt: %s", rephrase_prompt)
        subqueries = self.llm.generate(rephrase_prompt).strip().splitlines()
        subqueries = [q.strip() for q in subqueries if q.strip()]
        self.logger.debug("Final subqueries: %s", subqueries)

        return subqueries
