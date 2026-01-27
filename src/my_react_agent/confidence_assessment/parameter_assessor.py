from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..llm_adapters.llm_base import LLMBase

from ..agent_prompts.prompts_ids import PromptId
from ..agent_prompts.prompt_registry import PromptRegistry

class ParameterAssessor(ABC):
    def __init__(self, llm: LLMBase, *, prompts: PromptRegistry, default_fallback: float = 0.5, log_parse_failures: bool = True):
        self.llm = llm
        self.prompts = prompts
        self.default_fallback = default_fallback
        self.log_parse_failures = log_parse_failures

    def _render_prompt(self, pid: PromptId, **vars: Any) -> str:
        return self.prompts.render(pid.value, **vars)

    @abstractmethod
    def assess(
        self,
        *,
        query_text: str,
        answer_text: str,
        tool_result_text: str = "",
        knowledge_cutoff: Optional[str] = None,
        result_timestamp: Optional[str] = None,
    ) -> ParameterRating:
        raise NotImplementedError
