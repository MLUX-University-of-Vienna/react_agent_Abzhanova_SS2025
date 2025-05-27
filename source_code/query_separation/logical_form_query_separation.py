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
        # 1) Prompting LLM to produce minimal 
        prompt = (
            "SYSTEM: You are a query parser. "
            "Extract every phrase in the form '<field> of <entity>' from the user query. "
            "For each phrase, output DSL tokens as field(entity), where spaces in field are replaced by underscores. "
            "Join tokens with ' ∧ ' (space-ampersand-space)."
            "For example:"
            "  - 'capital of France' → capital(France)"
            "  - 'population density of France' → population_density(France)"
            "Respond *only* with the DSL, without any extra text."
            f"User query: '{query}'"
        )
        self.logger.debug("Separation prompt:%s", prompt)
        logic = self.llm.generate(prompt).strip()
        (prompt).strip()
        self.logger.debug("Received logic DSL: %s", logic)

        # 2) Split into segments
        segments = [seg.strip() for seg in logic.split('∧') if seg.strip()]
        subqueries: List[str] = []
        for seg in segments:
            # Expect format: field(entity)
            if '(' in seg and seg.endswith(')'):
                field, rest = seg.split('(', 1)
                entity = rest[:-1]  # drop trailing ')'
                human_field = field.replace('_', ' ').strip()
                human_entity = entity.strip()
                subqueries.append(f"What is the {human_field} of {human_entity}?" )
            else:
                self.logger.warning("Unparsable DSL segment: %s", seg)

        if not subqueries:
            self.logger.info("Falling back to simple punctuation split.")
            return [part.strip() for part in re.split(r'[,.] ', query) if part.strip()]

        return subqueries
