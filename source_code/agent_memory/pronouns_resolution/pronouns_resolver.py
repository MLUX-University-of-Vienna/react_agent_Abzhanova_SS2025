import re
import logging
from typing import List, Optional
from .entity_extractor import EntityExtractor
import gender_guesser.detector as gender

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

class PronounsResolver:
    def __init__(self, extractor: EntityExtractor):
        self.extractor = extractor
        self.context = {
            "male": None,
            "female": None,
            "neutral": None,
            "last": None,
            "last_answer": None,
            "last_numeric": None,
            "numeric_candidates": []
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

            entity = self.extractor.extract(sq)
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
        replacements = {
            r'\bhe\b': self.context.get("male"),
            r'\bshe\b': self.context.get("female"),
            r'\bit\b': self.context.get("last_numeric") or self.context.get("neutral"),
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

    def _update_context_from_answer(self, answer: str):
        self.context["last_answer"] = answer

        if "numeric_candidates" not in self.context:
            self.context["numeric_candidates"] = []

        numbers = re.findall(r'\b\d{1,4}\b', answer)
        for num in numbers:
            num = num.strip()
            if not num:
                continue

            score = 0
            lowered = answer.lower()
            if re.fullmatch(r'\d+', answer.strip()):
                score += 3  # pure number, likely direct answer
            if "age" in lowered or "years old" in lowered:
                score += 3
            if "born" in lowered or "birth" in lowered or "year" in lowered:
                score -= 1  # less relevant than direct age
            if int(num) < 1900 or int(num) > 2100:
                score += 1  # likely an actual answer, not a year

            # Save with score and answer context
            self.context["numeric_candidates"].append((num, score, answer.strip()))

        # Keep top scoring one
        if self.context["numeric_candidates"]:
            sorted_candidates = sorted(self.context["numeric_candidates"], key=lambda x: -x[1])
            best_num = sorted_candidates[0][0]
            self.context["last_numeric"] = best_num
            logger.debug(f"[Smart resolver] Selected numeric: {best_num} (from: '{sorted_candidates[0][2]}')")


    def _guess_gender(self, entity: str) -> Optional[str]:
        if re.search(r'\b(president|king|man|Mr\.)\b', entity, re.IGNORECASE):
            return "male"
        if re.search(r'\b(queen|woman|Mrs\.|Ms\.)\b', entity, re.IGNORECASE):
            return "female"
        return "neutral"

    def guess_gender(name: str) -> str:
        gender_detector = gender.Detector()
        first_name = name.split()[0] 
        g = gender_detector.get_gender(first_name)
        if g in ["male", "mostly_male"]:
            return "male"
        elif g in ["female", "mostly_female"]:
            return "female"
        return "neutral"