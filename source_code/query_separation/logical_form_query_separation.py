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
            "SYSTEM: You are a query parser. "
            "Extract all '<field> of <entity>'-style phrases from the user's query.\n"
            "Return each as DSL in the form field(entity), replacing spaces with underscores but preserving determiners like 'the'.\n"
            "Join tokens with ' ∧ ' (space-ampersand-space)."
            "For example:"
            "  - 'capital of France' → capital(France)"
            "  - 'population density of Netherlands' → population_density(Netherlands)"
            f"User query: '{query}'"
            "Return only the DSL string:"
        )
        self.logger.debug("Separation prompt:%s", prompt)
        logic = self.llm.generate(prompt).strip()
        self.logger.debug("Received logic DSL: %s", logic)

        # 2) Split into segments
        segments = [seg.strip() for seg in logic.split('∧') if seg.strip()]

        if not segments:
            self.logger.warning("No valid DSL segments found; falling back to punctuation-based splitting.")
            return [part.strip() for part in re.split(r'[,.]', query) if part.strip()]

        rephrase_prompt = (
            "SYSTEM: You are a helpful assistant.\n"
            "Convert the following DSL expressions into clear natural-language sub-questions.\n"
            "Each DSL is in the format: field(entity)\n"
            "Respond with one question per line.\n"
            f"DSL: {' ∧ '.join(segments)}"
        )

        self.logger.debug("Rephrasing prompt: %s", rephrase_prompt)
        subqueries = self.llm.generate(rephrase_prompt).strip().splitlines()
        subqueries = [q.strip() for q in subqueries if q.strip()]
        self.logger.debug("Final subqueries: %s", subqueries)

        return subqueries
