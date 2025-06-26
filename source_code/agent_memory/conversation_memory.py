from typing import List, Dict
import logging
import re

from .pronouns_resolver import PronounsResolver

logger = logging.getLogger(__name__)

class ConversationMemory:
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.resolver = PronounsResolver()

    def update(self, query: str, answer: str):
        self.history.append({"query": query, "answer": answer})
        logger.debug(f"Updated conversation memory. Current length: {len(self.history)}")

    def get_relevant_context(self) -> str:#
        if not self.history:
            return ""
        context = "\n".join([f"Q: {entry['query']}\nA: {entry['answer']}" for entry in self.history[-5:]])
        return f"Recent conversation context:\n{context}"

    def resolve_pronouns(self, text: str) -> str:
        prior_answers = [entry["answer"] for entry in self.history[-3:]]
        resolved = self.resolver.resolve(["", query], prior_answers=prior_answers)
        return resolved[-1]

    