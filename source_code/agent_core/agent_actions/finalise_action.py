import logging
from typing import List, Optional
from .action import Action

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FinaliseAction(Action):
    def __init__(self) -> None:
        self._cached_name_value = "FINALISE"
        logger.info("[FinaliseAction.__init__] Initialised action with evidence-led completion policy")

    @property
    def name(self) -> str:
        logger.debug(f"[FinaliseAction.name] Returning name value length={len(self._cached_name_value)}")
        return self._cached_name_value

    @property
    def instructions(self) -> str:
        value = (
            "Produce the final answer in British English using gathered evidence and clearly stated reasoning. "
            "Do not trigger further tools. "
            "Be direct, include essential qualifiers if uncertainty remains, and avoid introducing any new unverifiable facts."
        )
        logger.debug(f"[FinaliseAction.instructions] Instructions length={len(value)} preview={value[:96]!r}")
        return value

    @property
    def when_to_pick(self) -> str:
        value = (
            "Pick this when all prerequisite facts are gathered or can be deterministically computed, "
            "the synthesis is coherent, and further tool usage would be redundant. "
            "If small gaps remain that do not change the outcome, acknowledge them briefly and complete the answer."
        )
        logger.debug(f"[FinaliseAction.when_to_pick] Guidance length={len(value)} preview={value[:96]!r}")
        return value

    @property
    def when_to_pick_head(self) -> Optional[List[str]]:
        items = [
            "All atomic fact slots are DONE or derivable from DONE values",
            "Any required calculation is DONE or trivial from available numbers",
            "Confidence exceeds the acceptance threshold for the task",
            "Additional tool calls would only restate known information",
            "The user indicates they want the final answer now",
        ]
        logger.debug(f"[FinaliseAction.when_to_pick_head] Items count={len(items)} head_preview={items[0]!r}")
        return items

    @property
    def examples(self) -> List[str]:
        items = [
            "compute the percentage after both populations have been retrieved",
            "summarise findings after reading the provided snippets",
            "conclude once all subqueries return single values",
            "final answer once the calculation is confirmed",
        ]
        logger.debug(f"[FinaliseAction.examples] Examples count={len(items)} first={items[0]!r}")
        return items
