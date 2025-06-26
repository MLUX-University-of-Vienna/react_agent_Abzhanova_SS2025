import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

class PronounsResolver:
    def __init__(self):
        self.context = {
            "male": None,
            "female": None,
            "neutral": None,
            "last": None,          
            "last_answer": None,  
            "last_numeric": None, 
        }

    def resolve(self, subqueries: List[str], prior_answers: Optional[List[str]] = None) -> List[str]:
        """Resolves pronouns in each subquery using tracked named entities or values from prior answers."""
        if not subqueries:
            logger.debug("No subqueries provided for resolution.")
            return subqueries

        if prior_answers:
            for ans in reversed(prior_answers):
                self._update_context_from_answer(ans)

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

            resolved.append(sq)

        logger.debug(f"Resolved subqueries: {resolved}")
        return resolved

    def _replace_pronouns(self, text: str) -> str:
        last_number = self.context.get("last_numeric")
        replacements = {
            r'\bhe\b': self.context.get("male"),
            r'\bshe\b': self.context.get("female"),
            r'\bit\b': last_number if last_number else self.context.get("neutral"),
            r'\bthey\b': self.context.get("last"),
            r'\bhis\b': self._possessive_form(self.context.get("male")),
            r'\bher\b': self._possessive_form(self.context.get("female")),
            r'\btheir\b': self._possessive_form(self.context.get("last")),
        }

        for pattern, replacement in replacements.items():
            if replacement:
                new_text = re.sub(pattern, str(replacement), text, flags=re.IGNORECASE)
                if new_text != text:
                    logger.debug(f"Replaced '{pattern}' with '{replacement}' in '{text}'")
                    text = new_text

        logger.debug(f"Final transformed query: '{text}'")
        return text


    def _possessive_form(self, entity: Optional[str]) -> Optional[str]:
        if not entity:
            return None
        return entity + "'" if entity.endswith("s") else entity + "'s"

    def _extract_named_entity(self, text: str) -> Optional[str]:
        match = re.search(
            r"(?:the\s)?(?:name of\s)?((?:president|king|queen|prime minister|CEO|founder|director) of [A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)",
            text, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()

        tokens = re.findall(r"\b[A-Z][a-z]+\b(?:\s[A-Z][a-z]+\b)*", text)
        for token in tokens:
            if token.lower() not in {"what", "how", "who", "where", "when", "why"}:
                return token.strip()

        return None

    def _update_context_from_answer(self, answer: str):
        self.context["last_answer"] = answer
        match = re.search(r'\b\d+(?:\.\d+)?\b', answer)
        if match:
            self.context["last_numeric"] = match.group(0)
            logger.debug(f"Extracted numeric value '{match.group(0)}' from answer for future reference.")

    def _guess_gender(self, entity: str) -> Optional[str]:
        if re.search(r'\b(president|king|man|Mr\.)\b', entity, re.IGNORECASE):
            return "male"
        if re.search(r'\b(queen|woman|Mrs\.|Ms\.)\b', entity, re.IGNORECASE):
            return "female"
        return "neutral"
