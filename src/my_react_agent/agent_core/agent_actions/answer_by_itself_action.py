from __future__ import annotations

from .action import Action


class AnswerByItselfAction(Action):
    @property
    def name(self) -> str:
        return "ANSWER_BY_ITSELF"

    @property
    def default_instructions(self) -> str:
        return (
            "Answers using only the LLM's general knowledge or when result is known from previous step. "
        )

    @property
    def default_when_to_pick(self) -> str:
        return (
            "Pick ONLY when the step can be answered from stable, textbook-level facts "
            "(very unlikely to change over 3+ years) or when result is known from previous step."
            "DO NOT pick if the step asks for any real-world data that may change."
        )
