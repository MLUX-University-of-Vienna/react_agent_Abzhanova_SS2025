from typing import List, Dict
import logging
import re

from agent_memory.pronouns_resolution.pronouns_resolver import PronounsResolver
from .entity_class import Entity

logger = logging.getLogger(__name__)

class ConversationMemory:
    def __init__(self, pronouns_resolver: PronounsResolver):
        self.history: List[Dict[str, Optional[Entity]]] = []
        self.pronouns_resolver = pronouns_resolver
        self.MAX_TURNS = 5

    def update(self, query: str, answer: str):
        query_entity: Optional[Entity] = self.pronouns_resolver.entity_extractor.extract(query)
        answer_entity: Optional[Entity] = self.pronouns_resolver.entity_extractor.extract(answer)
        self.history.append({
            "query":        query,
            "query_entity":    query_entity,
            "answer":       answer,
            "answer_entity":   answer_entity,
        })
        logger.debug(f"Updated conversation memory. Current length: {len(self.history)}")

    def get_relevant_context(self) -> str:
        if not self.history:
            return ""
        lines = [
            f"Q: {e['query']}\nA: {e['answer']}"
            for e in self.history[-5:]
        ]
        return "Recent conversation context:\n" + "\n".join(lines)

    def resolve_pronouns(self, text: str) -> str:
        prior_answers = [entry["answer"] for entry in self.history]
        query = self.history[0]["query"] if self.history else ""
        resolved = self.pronouns_resolver.resolve(
            [query, text],
            prior_answers=prior_answers
        )
        return resolved[-1]

    def _find_entity(self, text: str, gender: str = None) -> str:
        matches = re.findall(r"([A-Z][a-z]+(?:\s[A-Z]\.)?(?:\s[A-Z][a-z]+)+)", text)
        if matches:
            return matches[0]  
        return ""

    