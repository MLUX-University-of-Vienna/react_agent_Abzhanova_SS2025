from __future__ import annotations

import json
import logging
from typing import Optional

from ..llm_adapters.llm_base import LLMBase

from .models import ParameterRating
from .parameter_assessor import ParameterAssessor
from .json_utils import _extract_json_object, _coerce_score_0_1, _pick_reason, _round2

logger = logging.getLogger(__name__)


class EntityAlignmentAssessor(ParameterAssessor):
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
        schema = {"same_entity": True, "confidence": 0.0, "reason": "short"}
        prompt = (
            "You are a strict evaluator for a QA system.\n\n"
            "Task: Decide whether the ANSWER is about the SAME REAL-WORLD ENTITY (or entities) as the QUESTION.\n"
            "Be careful with similarly-named words.\n"
            "If the answer is about a different entity with the same name, that is NOT the same entity.\n"
            "If the requested fact is missing, give 0. \n\n"
            "Output rules (CRITICAL):\n"
            "- Output MUST be a SINGLE JSON object and NOTHING else.\n"
            "- Keys MUST be exactly: same_entity, confidence, reason.\n"
            "- confidence MUST be one of: 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0\n"
            "- Do NOT include any extra keys.\n\n"
            f"Schema example: {json.dumps(schema)}\n\n"
            f"QUESTION:\n{query_text}\n\n"
            f"ANSWER:\n{answer_text}\n\n"
            "JSON:"
        )

        same_entity = False
        conf = self.default_fallback
        reason = "fallback"
        raw = ""

        try:
            raw = (self.llm.generate(prompt) or "").strip()
            obj = _extract_json_object(raw) or {}
            if not obj and self.log_parse_failures:
                logger.warning("[EntityAlignmentAssessor] JSON parse failed raw=%r", raw[:600])

            same_entity = bool(obj.get("same_entity"))
            conf = _coerce_score_0_1(obj.get("confidence", conf), conf)
            reason = _pick_reason(obj, reason)
        except Exception as e:
            logger.warning("[EntityAlignmentAssessor] LLM failed error=%r raw=%r", e, raw[:500])

        score = conf if same_entity else 0.0

        return ParameterRating(
            name="entity_alignment",
            score=_round2(score),
            reason=reason,
            meta={"same_entity": same_entity, "raw_confidence": _round2(conf)},
        )