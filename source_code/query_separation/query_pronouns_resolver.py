import re
import logger
from typing import List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

class QueryPronounsResolver:
    def resolve(self, subqueries: List[str]) -> List[str]:
        """Resolves pronouns in each subquery based on the nearest named entity found in previous subqueries"""
        if not subqueries:
            return subqueries

        resolved = []
        known_entities = []

        for i, sq in enumerate(subqueries):
            if i > 0:
                entity = self._last_valid_entity(known_entities)
                sq = self._replace_pronouns(sq, entity)

            entity = self._extract_named_entity(sq)
            if entity:
                known_entities.append(entity)

            resolved.append(sq)

        return resolved

    def _replace_pronouns(self, text: str, entity: str) -> str:
        if not entity:
            return text
        return re.sub(r'\b(he|she|they|it|his|her|their)\b', entity, text, flags=re.IGNORECASE)

    def _extract_named_entity(self, text: str) -> str:
        # Extracts first occurrence of a capitalized name-like phrase
        matches = re.findall(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
        return matches[0] if matches else ""
    
    def _last_valid_entity(self, entities: List[str]) -> str:
        for entity in reversed(entities):
            if not re.match(r"(?i)^what|who|where|when|why|how$", entity):
                return entity
        return ""
