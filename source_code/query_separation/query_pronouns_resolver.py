import re
from typing import List

class QueryPronounsResolver:
    def resolve(self, subqueries: List[str]) -> List[str]:
        if not subqueries:
            return subqueries

        resolved = []
        known_entities = []

         for i, sq in enumerate(subqueries):
            if i > 0:
                entity = known_entities[-1] if known_entities else ""
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