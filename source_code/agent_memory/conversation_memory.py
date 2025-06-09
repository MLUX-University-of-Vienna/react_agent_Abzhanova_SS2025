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

    def get_relevant_context(self) -> str:
        
    