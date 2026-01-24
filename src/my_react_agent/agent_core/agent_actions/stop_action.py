from __future__ import annotations

from .action import Action


class StopAction(Action):
    @property
    def name(self) -> str:
        return "STOP"

    @property
    def default_instructions(self) -> str:
        return (
            "End the session immediately and return a short acknowledgement. "
            "Use only when the user explicitly cancels, when a hard boundary is reached, "
            "or when repeated failures prevent progress and no safe recovery path exists."
        )

    @property
    def default_when_to_pick(self) -> str:
        return (
            "Pick when >3 times was choosen the same action with same input."
        )   
