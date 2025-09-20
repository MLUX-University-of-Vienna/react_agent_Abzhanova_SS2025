# actions/answer_by_itself_action.py
import logging
from typing import List, Optional
from .action import Action

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AnswerByItselfAction(Action):
    def __init__(self) -> None:
        self._cached_name_value = "ANSWER_BY_ITSELF"
        logger.info("[AnswerByItselfAction.__init__] Initialised with strict knowledge-only boundaries and conservative applicability")

    @property
    def name(self) -> str:
        logger.debug(f"[AnswerByItselfAction.name] Returning cached action name value length={len(self._cached_name_value)}")
        return self._cached_name_value

    @property
    def instructions(self) -> str:
        value = (
            "Answer directly only when the response is purely conceptual, non-numeric, non-comparative, and stable over time. "
            "Use this action solely for brief explanations of universally accepted concepts where no figures, dates, counts, "
            "ratios, or external validation are required. Do not infer quantities or perform calculations. "
            "If the user’s request contains or implies numbers, percentages, ratios, shares, comparisons between entities, "
            "time sensitivity, or any data retrieval, do not pick this action."
        )
        logger.debug(f"[AnswerByItselfAction.instructions] Instructions length={len(value)} preview={value[:120]!r}")
        return value

    @property
    def when_to_pick(self) -> str:
        value = (
            "Choose only when the task is a short, timeless description or definition of a single concept and does not require "
            "any lookup, citation, computation, measurement, timeline, or cross-entity comparison. "
            "Avoid this action if the user asks for anything quantitative, up-to-date, multi-step, or verifiable."
        )
        logger.debug(f"[AnswerByItselfAction.when_to_pick] Guidance length={len(value)} preview={value[:120]!r}")
        return value

    @property
    def when_to_pick_head(self) -> Optional[List[str]]:
        items = [
            "Single-concept, stable explanation without figures or dates.",
            "Brief meaning or role of a concept without metrics or examples.",
            "High-level mechanism overview where precision and recency are irrelevant.",
            "No tools, browsing, disambiguation, or calculations are necessary.",
        ]
        logger.debug(f"[AnswerByItselfAction.when_to_pick_head] Items count={len(items)} head_preview={items[0]!r}")
        return items

    @property
    def regex_patterns(self) -> Optional[List[str]]:
        patterns = [
            r"\d",
            r"\bpercent(?:age)?\b",
            r"\b%(\s|$)",
            r"\bratio\b|\bshare\b|\bproportion\b",
            r"\bcompared to\b|\brelative to\b|\bversus\b|\bvs\.?\b",
            r"\bpopulation\b|\bprice\b|\brate\b|\bgdp\b|\barea\b|\bcount\b|\btotal\b|\bnumber\b",
            r"\btoday\b|\bnow\b|\blatest\b|\brecent\b|\bthis year\b|\blast year\b|\bupdate\b",
            r"\bcalculate\b|\bcompute\b|\bestimate\b",
            r"\bbetween\b.+\band\b",
        ]
        logger.debug(f"[AnswerByItselfAction.regex_patterns] Patterns count={len(patterns)}")
        return patterns

    @property
    def examples(self) -> List[str]:
        items = [
            "brief conceptual definition (no figures, no dates)",
            "short explanation of a widely accepted concept (no metrics)",
            "outline of a mechanism in one sentence (no numbers)",
            "concise meaning of a term (single entity, timeless)",
        ]
        logger.debug(f"[AnswerByItselfAction.examples] Examples count={len(items)} first={items[0]!r}")
        return items
