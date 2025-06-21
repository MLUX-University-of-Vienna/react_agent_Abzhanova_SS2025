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

    
    def _extract_named_entity(self, text: str) -> str:
        # Extracts first occurrence of a capitalized name-like phrase
        matches = re.findall(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
        return matches[0] if matches else ""