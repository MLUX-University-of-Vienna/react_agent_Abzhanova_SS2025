import logging
from typing import List, Optional
from .action import Action

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StopAction(Action):
    def __init__(self) -> None:
        self._cached_name_value = "STOP"
        logger.info("[StopAction.__init__] Initialised action with deterministic name and conservative termination policy")

    @property
    def name(self) -> str:
        logger.debug(f"[StopAction.name] Returning name value length={len(self._cached_name_value)}")
        return self._cached_name_value

    @property
    def instructions(self) -> str:
        value = (
            "End the session immediately and return a short acknowledgement. "
            "Use only when the user explicitly cancels, when a hard safety or permission boundary is reached, "
            "or when further steps would repeat without progress and no safe recovery path exists."
        )
        logger.debug(f"[StopAction.instructions] Instructions length={len(value)} preview={value[:96]!r}")
        return value

    @property
    def when_to_pick(self) -> str:
        value = (
            "Pick this when the user requests to stop, cancel, or end; when policy prohibits continuation; "
            "when required authorisation or non-public data is unavailable; when repeated failures or rate limits prevent progress; "
            "or when the best course is to cease rather than hallucinate a result."
        )
        logger.debug(f"[StopAction.when_to_pick] Guidance length={len(value)} preview={value[:96]!r}")
        return value

    @property
    def when_to_pick_head(self) -> Optional[List[str]]:
        items = [
            "Explicit stop, cancel, abort, quit, or end requests by the user",
            "Irrecoverable safety, legal, or permission boundary",
            "Non-public or unavailable credentials required to proceed",
            "Detected infinite loop, duplicate action, or repeated non-actionable error",
            "Third-party rate limits or outages with no reasonable fallback",
        ]
        logger.debug(f"[StopAction.when_to_pick_head] Items count={len(items)} head_preview={items[0]!r}")
        return items

    @property
    def examples(self) -> List[str]:
        items = [
            "stop now",
            "that is enough, end here",
            "cancel this task",
            "never mind, please quit",
            "forget it, terminate the session",
        ]
        logger.debug(f"[StopAction.examples] Examples count={len(items)} first={items[0]!r}")
        return items
