from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .answer_generator import AnswerGenerator
from tool_management.tool_executor import ToolExecutor
from tool_management.tool_query_refiner import ToolQueryRefiner
from confidence_assessment.confidence_assessor import ConfidenceAssessor


@dataclass
class ActionServices:
    answer_generator: AnswerGenerator
    tool_executor: ToolExecutor
    tool_query_refiner: ToolQueryRefiner
    confidence_assessor: Optional[ConfidenceAssessor] = None
