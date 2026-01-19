from __future__ import annotations

from agent_core.agent_actions.action import Action


class AnswerByItselfAction(Action):
    @property
    def name(self) -> str:
        return "ANSWER_BY_ITSELF"

    @property
    def instructions(self) -> str:
        return (
            "Answers using only the LLM's general knowledge. "
        )

    @property
    def when_to_pick(self) -> str:
        return (
            "Pick ONLY when the step can be answered from stable, textbook-level facts "
            "(very unlikely to change over 3+ years)."
            "DO NOT pick if the step asks for any real-world data that may change."
        )
