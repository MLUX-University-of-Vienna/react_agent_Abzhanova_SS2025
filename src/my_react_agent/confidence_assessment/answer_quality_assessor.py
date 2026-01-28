from __future__ import annotations

import json
import logging
from typing import Optional

from ..llm_adapters.llm_base import LLMBase
from ..agent_prompts.defaults_prompts import DEFAULT_PROMPTS
from ..agent_prompts.prompt_template import PromptTemplate
from ..agent_prompts.prompts_ids import PromptId
from ..agent_prompts.prompt_registry import PromptRegistry

from .json_utils import _coerce_score_0_1, _extract_json_object, _pick_reason, _round2
from .models import ParameterRating
from .parameter_assessor import ParameterAssessor

logger = logging.getLogger(__name__)


class AnswerQualityAssessor(ParameterAssessor):
    def __init__(
        self,
        llm: LLMBase,
        *,
        prompts: PromptRegistry,
        default_fallback: float = 0.5,
        log_parse_failures: bool = True,
    ):
        super().__init__(llm, prompts=prompts, default_fallback=default_fallback, log_parse_failures=log_parse_failures)
        logger.info(
            "[AnswerQualityAssessor.__init__] ready prompt_id=%s fallback=%.2f",
            self.prompt_id,
            self.default_fallback,
        )

    @property
    def name(self) -> str:
        return "answer_quality"

    @property
    def prompt_id(self) -> str:
        return PromptId.CONF_ANSWER_QUALITY.value

    def default_prompt_template(self) -> PromptTemplate:
        tpl = DEFAULT_PROMPTS.get(PromptId.CONF_ANSWER_QUALITY) or DEFAULT_PROMPTS.get(self.prompt_id)
        if tpl is None:
            raise KeyError(f"DEFAULT_PROMPTS missing template for {PromptId.CONF_ANSWER_QUALITY!r} / {self.prompt_id!r}")
        return tpl

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

        logger.debug(
            "[AnswerQualityAssessor.assess] q_len=%d a_len=%d tool_len=%d",
            len(query_text or ""),
            len(answer_text or ""),
            len(tool_result_text or ""),
        )

        prompt = self._render_prompt(
            schema_example=json.dumps(schema),
            knowledge_cutoff_block=knowledge_cutoff or "",
            result_timestamp_block=result_timestamp or "",
            question=query_text,
            answer=answer_text,
            tool_result_block=tool_result_text or "",
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

        out = ParameterRating(
            name=self.name,
            score=_round2(score),
            reason=reason,
            meta={},
        )
        logger.debug("[AnswerQualityAssessor.assess] score=%.2f reason=%s", out.score, out.reason)
        return out