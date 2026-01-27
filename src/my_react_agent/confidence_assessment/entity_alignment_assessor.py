from __future__ import annotations

import json
import logging
from typing import Optional

from ..llm_adapters.llm_base import LLMBase

from .models import ParameterRating
from .parameter_assessor import ParameterAssessor
from .json_utils import _extract_json_object, _coerce_score_0_1, _pick_reason, _round2

from ..agent_prompts.prompts_ids import PromptId
from ..agent_prompts.prompt_registry import PromptRegistry

logger = logging.getLogger(__name__)

class EntityAlignmentAssessor(ParameterAssessor):
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
        schema = {"same_entity": True, "confidence": 0.0, "reason": "short"}
        prompt = self._render_prompt(
            PromptId.CONF_ENTITY_ALIGNMENT,
            schema_json=json.dumps(schema),
            query_text=query_text,
            answer_text=answer_text,
            tool_result_text=tool_result_text or "",
            knowledge_cutoff=knowledge_cutoff or "",
            result_timestamp=result_timestamp or "",
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