from __future__ import annotations

import json
import logging
from typing import Optional

from ..llm_adapters.llm_base import LLMBase

from .models import LLMAnswerQuality
from .parameter_assessor import ParameterAssessor
from .json_utils import _extract_json_object, _coerce_score_0_1, _pick_reason, _round2

logger = logging.getLogger(__name__)


class AnswerQualityAssessor(ParameterAssessor):
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
    ) -> ParameterRating:
        schema = {"score": 0.0, "reason": "short"}
        kc = f"Knowledge cutoff: {knowledge_cutoff}\n" if knowledge_cutoff else ""
        ts = f"Tool result timestamp: {result_timestamp}\n" if result_timestamp else ""
        tool_block = f"\n\nTOOL RESULT:\n{tool_result_text}\n" if tool_result_text else ""

        prompt = (
            "You are an evaluator for a QA system.\n\n"
            "Task: Score how completely the ANSWER provides the facts needed to answer the QUESTION.\n\n"
            "CRITICAL guidance:\n"
            "- The ANSWER may be long, messy, or poorly structured. That is OK.\n"
            "- The MOST IMPORTANT thing is whether the ANSWER CONTAINS the correct facts needed to answer the QUESTION.\n"
            "Score meaning:\n"
            "- 1.0 = contains the key facts needed to answer (even if verbose/messy).\n"
            "- 0.5 = contains some relevant facts but misses key facts.\n"
            "- 0.0 = the ANSWER does not contain requested fact / presents the lack of information / when real data is needed, but the ANSWER is outdated.\n\n"
            "Rules:\n"
            "- If the TOOL RESULT contradicts the answer, lower the score.\n"
            "- If the TOOL RESULT is irrelevant/empty, ignore it and judge using the question/answer only.\n"
            "- Do NOT reward confident tone; reward correctness and presence of needed facts.\n\n"
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
