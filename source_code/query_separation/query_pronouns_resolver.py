import re
from typing import List

class QueryPronounsResolver:
    def resolve(self, subqueries: List[str]) -> List[str]:
        if not subqueries:
            return subqueries

        first_entity = self._extract_named_entity(subqueries[0])
        if not first_entity:
            return subqueries

    
    def _extract_named_entity(self, text: str) -> str:
        # Extracts first occurrence of a capitalized name-like phrase
        matches = re.findall(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
        return matches[0] if matches else ""