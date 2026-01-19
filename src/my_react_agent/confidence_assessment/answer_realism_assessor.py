from __future__ import annotations

import json
import logging
from typing import Optional

from ..llm_adapters.llm_base import LLMBase

from .models import LLMAnswerRealism
from .parameter_assessor import ParameterAssessor
from .json_utils import _extract_json_object, _coerce_score_0_1, _pick_reason, _round2

logger = logging.getLogger(__name__)


class AnswerRealismAssessor(ParameterAssessor):
    def __init__(self, llm: LLMBase, *, default_fallback: float = 0.5, log_parse_failures: bool = True):
        super().__init__(llm, default_fallback=default_fallback, log_parse_failures=log_parse_failures)

    def assess(
        self,
        *,
        query_text: str,
        answer_text: str,
        tool_result_text: str = "",
        knowledge_cutoff: Optional[str] = None,
        result_timestamp: Optional[str] = None,
    ) -> LLMAnswerRealism:
        schema = {"score": 0.0, "reason": "short"}
        kc = f"Knowledge cutoff: {knowledge_cutoff}\n" if knowledge_cutoff else ""
        ts = f"Tool result timestamp: {result_timestamp}\n" if result_timestamp else ""
        tool_block = f"\n\nTOOL RESULT:\n{tool_result_text}\n" if tool_result_text else ""

        prompt = (
            "You are an evaluator for a QA system.\n\n"
            "Task: Assess REALISM of the ANSWER according to general world knowledge.\n"
            "Realism means: the answer is plausible, internally consistent, and not obviously fabricated.\n\n"
            "Important:\n"
            "- The answer may be long or messy; judge plausibility of the stated facts, not writing quality.\n\n"
            "Scoring:\n"
            "- 1.0 = very plausible for the question.\n"
            "- 0.5 = somewhat plausible but has mild inconsistencies. \n"
            "- 0.0 = implausible / likely fabricated / not useful / the requested fact is missing.\n\n"
            "Output rules (CRITICAL):\n"
            "- Output MUST be a SINGLE JSON object and NOTHING else.\n"
            "- Keys MUST be exactly: score, reason.\n"
            "- score MUST be one of: 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0\n"
            "- Do NOT include any extra keys.\n"
            "- reason MUST be a short string (<= 20 words).\n\n"
            f"Schema example: {json.dumps(schema)}\n\n"
            f"{kc}{ts}"
            f"QUESTION:\n{query_text}\n\n"
            f"ANSWER:\n{answer_text}"
            f"{tool_block}\n"
            "JSON:"
        )

        score = self.default_fallback
        reason = "fallback"
        raw = ""

        try:
            raw = (self.llm.generate(prompt) or "").strip()
            obj = _extract_json_object(raw) or {}
            if not obj and self.log_parse_failures:
                logger.warning("[AnswerRealismAssessor] JSON parse failed raw=%r", raw[:600])

            score = _coerce_score_0_1(obj.get("score", score), score)
            reason = _pick_reason(obj, reason)
        except Exception as e:
            logger.warning("[AnswerRealismAssessor] LLM failed error=%r raw=%r", e, raw[:500])

        return LLMAnswerRealism(score=_round2(score), reason=reason)
