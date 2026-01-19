from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..llm_adapters.llm_base import LLMBase


class ParameterAssessor(ABC):
    def __init__(
        self,
        llm: LLMBase,
        *,
        default_fallback: float = 0.5,
        log_parse_failures: bool = True,
    ):
        self.llm = llm
        self.default_fallback = float(default_fallback)
        self.log_parse_failures = bool(log_parse_failures)

    @abstractmethod
    def assess(
        self,
        *,
        query_text: str,
        answer_text: str,
        tool_result_text: str = "",
        knowledge_cutoff: Optional[str] = None,
        result_timestamp: Optional[str] = None,
    ) -> Any:
        raise NotImplementedError
