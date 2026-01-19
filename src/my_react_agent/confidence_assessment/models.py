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
class LLMEntityAlignment:
    same_entity: bool
    confidence: float
    reason: str


@dataclasses.dataclass(frozen=True)
class LLMAnswerQuality:
    score: float  
    reason: str


@dataclasses.dataclass(frozen=True)
class LLMAnswerRealism:
    score: float  
    reason: str
