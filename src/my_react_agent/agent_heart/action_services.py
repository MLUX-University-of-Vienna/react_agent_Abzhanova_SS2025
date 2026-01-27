from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..tool_management.tool_executor import ToolExecutor
from ..tool_management.tool_query_refiner import ToolQueryRefiner
from ..confidence_assessment.confidence_assessor import ConfidenceAssessor


@dataclass
class ActionServices:
    tool_executor: ToolExecutor
    tool_query_refiner: ToolQueryRefiner
    confidence_assessor: Optional[ConfidenceAssessor] = None
