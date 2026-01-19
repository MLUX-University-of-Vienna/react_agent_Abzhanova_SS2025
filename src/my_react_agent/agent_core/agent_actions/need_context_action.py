from __future__ import annotations

from .action import Action


class NeedContextAction(Action):
    @property
    def name(self) -> str:
        return "RESOLVE_PRONOUNS_AND_OMITTED ENTITIES"

    @property
    def instructions(self) -> str:
        return (
            "Resolves pronouns and omitted entities in the CURRENT STEP to understand the context. "
            "Rewrites the step task into an explicit, self-contained instruction by replacing pronouns with entities. "
        )

    @property
    def when_to_pick(self) -> str:
        return (
            "Pick if the context of the CURRENT STEP is unclear because it contains pronouns and/or omitted entities. "
            "Do NOT pick if the CURRENT STEP's task is to define/determine/resolve those pronouns or omitted entities."
        )

