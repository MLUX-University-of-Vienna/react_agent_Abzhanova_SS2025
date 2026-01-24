from __future__ import annotations

import dataclasses

@dataclasses.dataclass(frozen=True)
class ToolResultAssessment:
    is_relevant: bool
    entity_aligned: bool
    has_required_numbers: bool  
    confidence: float
    reason: str
    supporting_text: str = ""


@dataclasses.dataclass(frozen=True)
class ParameterRating:
    name: str
    score: float
    reason: str
    meta: dict = dataclasses.field(default_factory=dict)
