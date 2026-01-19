from __future__ import annotations

from .action import Action


class UseToolAction(Action):
    @property
    def name(self) -> str:
        return "USE_TOOL"

    @property
    def instructions(self) -> str:
        return (
            "Invoke exactly ONE tool to complete the current step.\n"
            "Choose the most appropriate tool based strictly on its description.\n"
            "Provide minimal deterministic tool_input.\n"
            "Do not bundle multiple entities/facts."
        )

    @property
    def when_to_pick(self) -> str:
        return "Pick when user asks about real time data and at least one available tool can reliably complete the current step in one deterministic call."
