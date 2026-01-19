from __future__ import annotations

import re
from typing import Optional

from .text_utils import TextUtilsMixin

class FormattingMixin(TextUtilsMixin):
    def _format_thought_line(self, index: int, text: str) -> str:
        t = self._sanitise_to_text(text) or "I am considering the best next step."
        if not re.match(r"^(i|i’m|i am|let me|i should|i will|i think|now i)\b", t.lower()):
            t = f"I’m thinking: {t}"
        return f"Thought {index}: {self._ensure_sentence(t)}"

    def _format_action_line_human(
        self,
        index: int,
        text_action: str,
        tool_name: str,
        tool_input: str,
        message: Optional[str] = None,
        final_answer: Optional[str] = None,
    ) -> str:
        a = (text_action or "").upper().strip()
        tn = (tool_name or "").strip()
        ti = self._sanitise_to_text(tool_input)

        if a == "USE_TOOL":
            if tn:
                text = f"I’ll use {tn} with: {ti or 'default input'}."
            else:
                text = f"I’ll use the most suitable tool with: {ti or 'default input'}."
            return f"Act {index}: {self._ensure_sentence(text)}"

        if a == "ANSWER_BY_ITSELF":
            text = f"I’ll answer this step directly: {ti or 'focused micro-task'}."
            return f"Act {index}: {self._ensure_sentence(text)}"

        if a == "CLARIFY":
            msg = self._ensure_sentence(message or "Could you clarify?")
            return f"Act {index}: I’ll ask you to clarify: {msg}"

        if a == "STOP":
            msg = self._ensure_sentence(message or "Stopping here.")
            return f"Act {index}: {msg}"

        return f"Act {index}: {self._ensure_sentence(a.title())}"

    def _format_observation_line(self, index: int, text: str) -> str:
        t = self._ensure_sentence(text or "No immediately useful observation.")
        return f"Observation {index}: {t}"
