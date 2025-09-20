# actions/clarify.py
from .action import Action
from typing import List

class ClarifyAction(Action):
    @property
    def name(self) -> str:
        return "CLARIFY"

    @property
    def instructions(self) -> str:
        return ("Ask only when you have tried all tools and they didn't provide reliable answer "
                "or you can't answer by yourself. If a missing item is a public fact, do not clarify. "
                "Ask one crisp question, single sentence, <=25 words, offer 2–4 inline options separated by "
                "slashes including 'other', mark the default with (default).")

    @property
    def when_to_pick(self) -> str:
        return "Use when you have tried all tools and you you can't answer by yourself."

    @property
    def when_to_pick_head(self) -> List[str]:
        return [
            "Metric or measure is unclear (count/percentage/ranking/absolute vs relative).",
            "Time window, recency, or date range is unspecified.",
            "Geography or scope is ambiguous (global/country/region/dataset subset).",
            "Version, source, or dataset preference is required.",
            "Units, currency, or output format must be chosen.",
            "Output length, style, or language preference is needed.",
            "Thresholds, budgets, limits, or constraints affect the answer.",
            "Private or account data access is necessary.",
            "Pronoun or entity reference cannot be resolved from context."
        ]

    @property
    def examples(self) -> List[str]:
        return [
            "Which metric: count/percentage/ranking (default)?",
            "Which time period: latest year (default)/last 5 years/other?",
            "Which geography: global (default)/country/region/other?",
            "Which currency: USD (default)/EUR/GBP/other?",
            "Which units: metric (default)/imperial/other?",
            "Depth of answer: brief (default)/medium/deep?",
            "Preferred source: official statistics (default)/World Bank/IMF/other?",
            "Time window: today (default)/this week/this month/other?",
            "File format: CSV (default)/XLSX/JSON/other?",
            "Which company do you mean: Apple Inc (default)/another Apple/other?",
            "Language: English (default)/Spanish/French/other?",
            "Constraint focus: accuracy (default)/speed/brevity/lowest cost?"
        ]
