from __future__ import annotations

import json
import logging
from typing import Optional

from ..llm_adapters.llm_base import LLMBase

from .models import LLMAnswerQuality
from .parameter_assessor import ParameterAssessor
from .json_utils import _extract_json_object, _coerce_score_0_1, _pick_reason, _round2

from ..agent_prompts.prompts_ids import PromptId
from ..agent_prompts.prompt_registry import PromptRegistry

logger = logging.getLogger(__name__)

class AnswerQualityAssessor(ParameterAssessor):
    def __init__(self, llm: LLMBase, *, prompts: PromptRegistry, default_fallback: float = 0.5, log_parse_failures: bool = True):
        super().__init__(llm, prompts=prompts, default_fallback=default_fallback, log_parse_failures=log_parse_failures)

    def assess(
        self,
        *,
        query_text: str,
        answer_text: str,
        tool_result_text: str = "",
        knowledge_cutoff: Optional[str] = None,
        result_timestamp: Optional[str] = None,
    ) -> ParameterRating:
        schema = {"score": 0.0, "reason": "short"}
        kc = f"Knowledge cutoff: {knowledge_cutoff}\n" if knowledge_cutoff else ""
        ts = f"Tool result timestamp: {result_timestamp}\n" if result_timestamp else ""
        tool_block = f"\n\nTOOL RESULT:\n{tool_result_text}\n" if tool_result_text else ""

        prompt = self._render_prompt(
            PromptId.CONF_ANSWER_QUALITY,
            schema_json=json.dumps(schema),
            query_text=query_text,
            answer_text=answer_text,
            tool_result_text=tool_result_text or "",
            knowledge_cutoff=knowledge_cutoff or "",
            result_timestamp=result_timestamp or "",
        )

        score = self.default_fallback
        reason = "fallback"
        raw = ""

        try:
            raw = (self.llm.generate(prompt) or "").strip()
            obj = _extract_json_object(raw) or {}
            if not obj and self.log_parse_failures:
                logger.warning("[AnswerQualityAssessor] JSON parse failed raw=%r", raw[:600])

            score = _coerce_score_0_1(obj.get("score", score), score)
            reason = _pick_reason(obj, reason)
        except Exception as e:
            logger.warning("[AnswerQualityAssessor] LLM failed error=%r raw=%r", e, raw[:500])

        return ParameterRating(
            name="answer_quality",
            score=_round2(score),
            reason=reason,
            meta={},
        )
