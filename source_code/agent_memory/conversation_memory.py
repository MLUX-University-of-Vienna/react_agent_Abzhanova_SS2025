from typing import List, Dict
import logging
import re

logger = logging.getLogger(__name__)

class ConversationMemory:
    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def update(self, query: str, answer: str):
        self.history.append({"query": query, "answer": answer})
        logger.debug(f"Updated conversation memory. Current length: {len(self.history)}")

    def get_relevant_context(self) -> str:#
        if not self.history:
            return ""
        context = "\n".join([f"Q: {entry['query']}\nA: {entry['answer']}" for entry in self.history[-5:]])
        return f"Recent conversation context:\n{context}"

    def resolve_pronouns(self, text: str) -> str:
        last_answers = " ".join([entry["answer"] for entry in reversed(self.history[-3:])])
        replacements = {
            "he": self._find_entity(last_answers, gender="male"),
            "she": self._find_entity(last_answers, gender="female"),
            "they": self._find_entity(last_answers),
        }
        for pronoun, entity in replacements.items():
            if entity:
                text = re.sub(rf"\b{pronoun}\b", entity, text, flags=re.IGNORECASE)
        return text

    def _find_entity(self, text: str, gender: str = None) -> str:
        matches = re.findall(r"([A-Z][a-z]+(?:\s[A-Z]\.)?(?:\s[A-Z][a-z]+)+)", text)
        if matches:
            return matches[0]  
        return ""
        

    