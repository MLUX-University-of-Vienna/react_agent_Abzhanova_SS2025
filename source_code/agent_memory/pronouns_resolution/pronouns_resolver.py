import re
import logging
from typing import List, Optional
from .entity_extractor import EntityExtractor
from .spacy_entity_extractor import SpacyEntityExtractor
import gender_guesser.detector as gender

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
 
class PronounsResolver:
    def __init__(self, extractor: EntityExtractor):
        self.extractor = extractor or SpacyEntityExtractor()
        self.gender_detector = gender.Detector()
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
        if not subqueries:
            logger.debug("No subqueries provided for resolution.")
            return subqueries

        if prior_answers:
            self._update_context_from_answers(prior_answers)

        return self._resolve_subqueries(subqueries)

    def _update_context_from_answers(self, answers: List[str]):
        for answer in reversed(answers):
            self._update_context_from_single_answer(answer)

    def _resolve_subqueries(self, subqueries: List[str]) -> List[str]:
        resolved = []

        for i, sq in enumerate(subqueries):
            logger.debug(f"Processing subquery {i}: '{sq}'")

            if i > 0:
                sq = self._replace_pronouns(sq)

            self._update_context_from_query(sq)
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

    def _update_context_from_query(self, query: str):
        entity = self.extractor.extract(query)
        if entity:
            logger.debug(f"Extracted entity: '{entity}'")
            gender = self._guess_gender(entity)
            if gender:
                self.context[gender] = entity
            self.context["last"] = entity

    def _update_context_from_single_answer(self, answer: str):
        self.context["last_answer"] = answer

        self._extract_entity_from_answer(answer)
        self._extract_numeric_from_answer(answer)

    def _extract_entity_from_answer(self, answer: str):
        entity = self.extractor.extract(answer)
        if entity:
            gender = self._guess_gender(entity)
            if gender:
                self.context[gender] = entity
            self.context["last"] = entity
            logger.debug(f"[Smart resolver] Extracted entity from answer: {entity} (gender: {gender})")

    def _extract_numeric_from_answer(self, answer: str):
        numbers = re.findall(r'\b\d{1,4}\b', answer)
        for num in numbers:
            self._score_numeric_candidate(num, answer)

        if self.context["numeric_candidates"]:
            self._select_best_numeric()

    def _score_numeric_candidate(self, num: str, answer: str):
        num = num.strip()
        if not num:
            return

        score = 0
        lowered = answer.lower()

        if re.fullmatch(r'\d+', num):
            score += 3
        if "age" in lowered or "years old" in lowered:
            score += 3
        if any(term in lowered for term in ("born", "birth", "year")):
            score -= 1
        if int(num) < 1900 or int(num) > 2100:
            score += 1

        self.context["numeric_candidates"].append((num, score, answer.strip()))

    def _select_best_numeric(self):
        sorted_candidates = sorted(self.context["numeric_candidates"], key=lambda x: -x[1])
        best_num = sorted_candidates[0][0]
        self.context["last_numeric"] = best_num
        logger.debug(f"[Smart resolver] Selected numeric: {best_num} (from: '{sorted_candidates[0][2]}')")

    def _possessive_form(self, entity: Optional[str]) -> Optional[str]:
        if not entity:
            return None
        return entity + "'" if entity.endswith("s") else entity + "'s"

    def _guess_gender(self, entity: str) -> Optional[str]:
        first_name = entity.split()[0]
        g = self.gender_detector.get_gender(first_name)
        if g in ("male", "mostly_male"):
            return "male"
        elif g in ("female", "mostly_female"):
            return "female"
        return "neutral"
