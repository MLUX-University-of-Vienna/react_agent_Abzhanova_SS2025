import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

class QueryPronounsResolver:
    def __init__(self):
        self.context = {
            "male": None,
            "female": None,
            "neutral": None,
            "last": None,
        }

    def resolve(self, subqueries: List[str]) -> List[str]:
        """Resolves pronouns in each subquery based on the nearest named entity found in previous subqueries"""
        if not subqueries:
            logger.debug("No subqueries provided for resolution.")
            return subqueries

        resolved = []

        for i, sq in enumerate(subqueries):
            logger.debug(f"Processing subquery {i}: '{sq}'")

            if i > 0:
                sq = self._replace_pronouns(sq)

            entity = self._extract_named_entity(sq)
            if entity:
                logger.debug(f"Extracted entity: '{entity}'")
                gender = self._guess_gender(entity)
                if gender:
                    self.context[gender] = entity
                self.context["last"] = entity
            else:
                logger.debug("No valid named entity found in subquery.")

            resolved.append(sq)

        logger.debug(f"Resolved subqueries: {resolved}")
        return resolved

    def _replace_pronouns(self, text: str, entity: str) -> str:
        replacements = {
            r'\bhe\b': self.context.get("male"),
            r'\bshe\b': self.context.get("female"),
            r'\bit\b': self.context.get("neutral"),
            r'\bthey\b': self.context.get("last"),
            r'\bhis\b': self._possessive_form(self.context.get("male")),
            r'\bher\b': self._possessive_form(self.context.get("female")),
            r'\btheir\b': self._possessive_form(self.context.get("last")),
        }

        for pattern, replacement in replacements.items():
            if replacement:
                new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                if new_text != text:
                    logger.debug(f"Replaced '{pattern}' with '{replacement}' in '{text}'")
                    text = new_text

        return text

    def _possessive_form(self, entity: Optional[str]) -> Optional[str]:
        if not entity:
            return None
        return entity + "'" if entity.endswith("s") else entity + "'s"

    def _extract_named_entity(self, text: str) -> str:
        match = re.search(
            r"(?:the\s)?(?:name of\s)?((?:president|king|queen|prime minister|CEO|founder|director) of [A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)",
            text, re.IGNORECASE
        )
        if match:
            entity = match.group(1).strip()
            return entity

        tokens = re.findall(r"\b[A-Z][a-z]+\b(?:\s[A-Z][a-z]+\b)*", text)
        for token in tokens:
            if token.lower() not in {"what", "how", "who", "where", "when", "why"}:
                return token.strip()
        
        return None
    
    def _guess_gender(self, entity: str) -> Optional[str]:
        if re.search(r'\bpresident|king|man|Mr\.\b', entity, re.IGNORECASE):
            return "male"
        if re.search(r'\bqueen|woman|Mrs\.|Ms\.\b', entity, re.IGNORECASE):
            return "female"
        return "neutral"