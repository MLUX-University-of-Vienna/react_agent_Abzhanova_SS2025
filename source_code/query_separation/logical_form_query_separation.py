import json
import logging
import re
from typing import List

from llm_adapters.llm_base import LLMBase
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

    def __init__(self, llm: LLMBase):
        self.llm = llm

    def separate(self, query: str) -> List[str]:
        dsl_prompt = (
            "SYSTEM: Convert the user's question into a logical representation using the following DSL:\n"
            "- Use field(entity) expressions, joined by logical operators: ∧ (and), ∨ (or), ¬ (not)\n"
            "- Be strict: only represent what is explicitly asked.\n"
            "- Do not infer or expand the question.\n"
            f"User query: '{query}'\n"
            "Output only DSL expressions — no bullets, markdown, or explanation:"
        )

        self.logger.debug("Separation prompt:%s", dsl_prompt)

        logic = self.llm.generate(dsl_prompt).strip()
        self.logger.debug("Received logic DSL: %s", logic)

        if not logic:
            self.logger.warning("Empty DSL output from LLM.")
            return [query]

        if re.search(r'\b(he|she|they|it|his|her|their)\b', logic, re.IGNORECASE):
            self.logger.warning("DSL may contain unresolved pronouns: %s", logic)

        segments = [seg.strip() for seg in logic.split('∧') if seg.strip()]
        if not segments:
            self.logger.warning("No valid DSL segments found; falling back to punctuation-based splitting.")
            return [part.strip() for part in re.split(r'[,.]', query) if part.strip()]

        dsl_block = "\n".join(segments)

        rephrase_prompt = (
            "SYSTEM: For each logical field(entity) pair, generate a clear, minimal natural-language question.\n"
            "- Each line in the DSL corresponds to one question.\n"
            "- Do not combine or merge them.\n"
            "- Keep output simple: one question per line, no numbering or formatting.\n"
            f"Original query: {query}\n\n"
            f"DSL:\n{dsl_block}\n"
            "Questions:"
        )

        self.logger.debug("Rephrasing prompt: %s", rephrase_prompt)

        subqueries = self.llm.generate(rephrase_prompt).strip().splitlines()
        subqueries = [q.strip() for q in subqueries if q.strip()]
        self.logger.debug("Final subqueries: %s", subqueries)

        if not subqueries:
            self.logger.warning("No subqueries were generated; falling back to full query.")
            return [query]

        return subqueries
