from .action import Action
from typing import List
import logging

class SplitQueryAction(Action):
    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.SplitQueryAction")
        self._logger.setLevel(logging.INFO)
        self._logger.info("[SplitQueryAction.__init__] Initialised SplitQueryAction with strong preference for multi-hop, numeric, and comparative tasks")
        self._logger.debug("[SplitQueryAction.__init__] Action is designed to steer planning towards staged retrieval then compute when tasks contain multiple unknowns or comparisons")

    @property
    def name(self) -> str:
        value = "SPLIT_QUERY"
        self._logger.debug(f"[SplitQueryAction.name] value={value}")
        return value

    @property
    def instructions(self) -> str:
        text = (
            "Produce an ordered, minimal plan of atomic subqueries followed, if required, by one compute step. "
            "Each subquery must isolate exactly one subject and exactly one value, answerable in a single deterministic lookup or tool call. "
            "Do not include tool names, vendor hints, sources, or site filters; keep each item source-agnostic and reusable. "
            "Reference prior subquery results as a,b,c… in the final compute item written as 'compute: <expression>'. "
            "Include normalisation items first whenever units, currency, dates, geographies, or definitions must be aligned. "
            "Favour this action whenever the request involves more than one unknown, any cross-entity comparison, any percentage or ratio, or any calculation whose operands are not yet known. "
            "Write each item as a crisp imperative or noun-phrase of 6–14 words, with no conjunctions, no pipes, and no multiple subjects. "
            "Stop once all prerequisite facts and the compute step are present; do not expand beyond what is necessary."
        )
        self._logger.debug(f"[SplitQueryAction.instructions] length={len(text)} preview={text[:140]!r}")
        return text

    @property
    def when_to_pick(self) -> str:
        text = (
            "Answer the question: does the query require more than one fact? If the answer is yes, choose SPLIT_QUERY action"
        )
        self._logger.debug(f"[SplitQueryAction.when_to_pick] length={len(text)}")
        return text

    @property
    def when_to_pick_head(self) -> List[str]:
        items = [
            "Multiple unknowns that cannot be answered in a single lookup.",
            "Any percentage, ratio, share, or difference between distinct subjects.",
            "Comparisons, benchmarking, or phrasing that implies X versus Y.",
            "Aggregation or composition across items or over time windows.",
            "Required normalisation of units, currency, dates, or definitions.",
            "A qualitative resolution step is needed before fetching attributes.",
            "Operands for a calculation are not yet observed as numerals.",
            "A final value must be computed from earlier lookups."
        ]
        self._logger.debug(f"[SplitQueryAction.when_to_pick_head] count={len(items)} items={items}")
        return items

    @property
    def examples(self) -> List[str]:
        items = [
            "plan to obtain value for <entity A> then value for <entity B> then compute",
            "plan to collect values over <time window> then compute derived statistic",
            "plan to fetch value for <entity> in <unit> then normalise unit then compute",
            "plan to get values for <entity> across <regions> then compute aggregate",
            "plan to resolve <ambiguous entity> then fetch requested attribute",
            "plan to obtain start and end values for <series> then compute change"
        ]
        self._logger.debug(f"[SplitQueryAction.examples] count={len(items)}")
        return items
