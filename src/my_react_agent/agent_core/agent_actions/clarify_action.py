from __future__ import annotations

from .action import Action


class ClarifyAction(Action):
    @property
    def name(self) -> str:
        return "CLARIFY"

    @property
    def instructions(self) -> str:
        return (
            "Ask ONE clarification question only when the step cannot be completed reliably.\n"
            "Single sentence, <=25 words, with 2–4 inline options separated by slashes."
        )

    @property
    def when_to_pick(self) -> str:
        return (
            "Pick when the question is not clear even after completed NEED_CONTEXT action."
        )   