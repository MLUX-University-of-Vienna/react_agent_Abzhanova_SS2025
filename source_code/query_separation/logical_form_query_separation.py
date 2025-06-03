import json
import logging
import re
from typing import List, Dict

from llm.llm_interface import LLMInterface
from .query_separation_strategy import QuerySeparationStrategy

class LLMLogicalSeparationStrategy(QuerySeparationStrategy):
    """
    Splits a complex query by having the LLM emit a minimal 'field(entity) ∧ ...' DSL,
    then converts each field-entity pair into a natural-language subtask.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) 
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def separate(self, query: str) -> List[str]:
        # 1) Prompting LLM to produce minimal DSL
        dsl_prompt = (
            "SYSTEM: Convert the user's question into a logical representation using the following DSL:\n"
            "- Use field(entity) expressions, joined by logical operators: ∧ (and), ∨ (or), ¬ (not)\n"
            "- Be strict: only represent what is explicitly asked.\n"
            "- Do not infer or expand the question.\n"
            "- Resolve **all** pronouns (e.g., he, she, it, they) to their **full referents**.\n"
            "- Ensure the DSL is self-contained and unambiguous.\n"
            f"User query: '{query}'\n"
            "Output only DSL expressions — no bullets, markdown, or explanation:"
        )

        self.logger.debug("Separation prompt:%s", dsl_prompt)

        logic = self.llm.generate(dsl_prompt).strip()
        self.logger.debug("Received logic DSL: %s", logic)

        if not logic:
            self.logger.warning("Empty DSL output from LLM.")
            return [query]

        rephrase_prompt = (
            "SYSTEM: Convert each logical expression of the form field(entity) into a clear and minimal natural-language question.\n"
            f"DSL: {logic}\n"
            "Questions:"
        )

        self.logger.debug("Rephrasing prompt: %s", rephrase_prompt)
        subqueries = self.llm.generate(rephrase_prompt).strip().splitlines()
        subqueries = [q.strip() for q in subqueries if q.strip()]
        self.logger.debug("Final subqueries: %s", subqueries)

        return subqueries
