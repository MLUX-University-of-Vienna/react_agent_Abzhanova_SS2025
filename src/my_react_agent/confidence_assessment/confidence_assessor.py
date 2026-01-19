from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..llm_adapters.llm_base import LLMBase

from .models import (
    LLMEntityAlignment,
    LLMAnswerQuality,
    LLMAnswerRealism,
)

from .json_utils import (
    _safe_text,
    _clamp01,
    _serialise_tool_result,
)

from .entity_alignment_assessor import EntityAlignmentAssessor
from .answer_quality_assessor import AnswerQualityAssessor
from .answer_realism_assessor import AnswerRealismAssessor

logger = logging.getLogger(__name__)


class ConfidenceAssessor:

    def __init__(
        self,
        llm: LLMBase,
        *,
        default_fallback: float = 0.5,
        max_decimals: int = 2,
        entity_mismatch_penalty: float = 0.98,
        include_tool_result_in_quality: bool = True,
        tool_result_max_chars: int = 12000,
        log_parse_failures: bool = True,
    ):
        self.llm = llm
        self.default_fallback = _clamp01(default_fallback)
        self.max_decimals = max(0, int(max_decimals))
        self.entity_mismatch_penalty = _clamp01(entity_mismatch_penalty)
        self.include_tool_result_in_quality = bool(include_tool_result_in_quality)
        self.tool_result_max_chars = max(0, int(tool_result_max_chars))
        self.log_parse_failures = bool(log_parse_failures)

        self.entity_alignment_assessor = EntityAlignmentAssessor(
            llm=self.llm,
            default_fallback=self.default_fallback,
            log_parse_failures=self.log_parse_failures,
        )
        self.answer_quality_assessor = AnswerQualityAssessor(
            llm=self.llm,
            default_fallback=self.default_fallback,
            log_parse_failures=self.log_parse_failures,
        )
        self.answer_realism_assessor = AnswerRealismAssessor(
            llm=self.llm,
            default_fallback=self.default_fallback,
            log_parse_failures=self.log_parse_failures,
        )

        logger.info(
            "[ConfidenceAssessor.__init__] fallback=%.2f max_decimals=%d mismatch_penalty=%.2f "
            "include_tool_result_in_quality=%s tool_result_max_chars=%d log_parse_failures=%s",
            self.default_fallback,
            self.max_decimals,
            self.entity_mismatch_penalty,
            self.include_tool_result_in_quality,
            self.tool_result_max_chars,
            self.log_parse_failures,
        )

    def assess_step_summary(
        self,
        *,
        step_task: str,
        step_summary_evidence: Optional[Any],
        knowledge_cutoff: Optional[str] = None,
        result_timestamp: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        task_text = _safe_text(step_task)

        summary_text = _serialise_tool_result(step_summary_evidence)
        if self.tool_result_max_chars and len(summary_text) > self.tool_result_max_chars:
            summary_text = summary_text[: self.tool_result_max_chars].rstrip() + "..."

        entity = self._llm_entity_alignment(query_text=task_text, answer_text=summary_text)

        quality = self._llm_answer_quality(
            query_text=task_text,
            answer_text=summary_text,
            tool_result_text="",
            knowledge_cutoff=knowledge_cutoff,
            result_timestamp=result_timestamp,
        )

        realism = self._llm_answer_realism(
            query_text=task_text,
            answer_text=summary_text,
            tool_result_text="",
            knowledge_cutoff=knowledge_cutoff,
            result_timestamp=result_timestamp,
        )

        if not entity.same_entity:
            conf = entity.confidence * (1.0 - self.entity_mismatch_penalty)
            out = self._round_and_clamp(conf)
            logger.info(
                "[ConfidenceAssessor.assess_step_summary] entity_same=False entity_conf=%.3f quality=%.3f realism=%.3f "
                "-> conf=%.3f (penalized) reason_entity=%s reason_quality=%s reason_realism=%s",
                entity.confidence,
                quality.score,
                realism.score,
                out,
                entity.reason,
                quality.reason,
                realism.reason,
            )
            return out

        conf = (entity.confidence + quality.score + realism.score) / 3.0
        out = self._round_and_clamp(conf)
        logger.info(
            "[ConfidenceAssessor.assess_step_summary] entity_same=True entity_conf=%.3f quality=%.3f realism=%.3f "
            "-> conf=%.3f reason_entity=%s reason_quality=%s reason_realism=%s",
            entity.confidence,
            quality.score,
            realism.score,
            out,
            entity.reason,
            quality.reason,
            realism.reason,
        )
        return out

    def assess(
        self,
        query: str,
        result: Optional[Any],
        answer: str,
        knowledge_cutoff: Optional[str] = None,
        result_timestamp: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        query_text = _safe_text(query)
        answer_text = _safe_text(answer)

        tool_text = _serialise_tool_result(result)
        if self.tool_result_max_chars and len(tool_text) > self.tool_result_max_chars:
            tool_text = tool_text[: self.tool_result_max_chars].rstrip() + "..."

        entity = self._llm_entity_alignment(query_text=query_text, answer_text=answer_text)

        quality = self._llm_answer_quality(
            query_text=query_text,
            answer_text=answer_text,
            tool_result_text=(tool_text if self.include_tool_result_in_quality else ""),
            knowledge_cutoff=knowledge_cutoff,
            result_timestamp=result_timestamp,
        )

        realism = self._llm_answer_realism(
            query_text=query_text,
            answer_text=answer_text,
            tool_result_text=(tool_text if self.include_tool_result_in_quality else ""),
            knowledge_cutoff=knowledge_cutoff,
            result_timestamp=result_timestamp,
        )

        if not entity.same_entity:
            conf = entity.confidence * (1.0 - self.entity_mismatch_penalty)
            out = self._round_and_clamp(conf)
            logger.info(
                "[ConfidenceAssessor.assess] entity_same=False entity_conf=%.3f quality=%.3f realism=%.3f "
                "-> conf=%.3f (penalized) reason_entity=%s reason_quality=%s reason_realism=%s",
                entity.confidence,
                quality.score,
                realism.score,
                out,
                entity.reason,
                quality.reason,
                realism.reason,
            )
            return out

        conf = (entity.confidence + quality.score + realism.score) / 3.0
        out = self._round_and_clamp(conf)
        logger.info(
            "[ConfidenceAssessor.assess] entity_same=True entity_conf=%.3f quality=%.3f realism=%.3f "
            "-> conf=%.3f reason_entity=%s reason_quality=%s reason_realism=%s",
            entity.confidence,
            quality.score,
            realism.score,
            out,
            entity.reason,
            quality.reason,
            realism.reason,
        )
        return out


    def _llm_entity_alignment(self, *, query_text: str, answer_text: str) -> LLMEntityAlignment:
        return self.entity_alignment_assessor.assess(query_text=query_text, answer_text=answer_text)

    def _llm_answer_quality(
        self,
        *,
        query_text: str,
        answer_text: str,
        tool_result_text: str,
        knowledge_cutoff: Optional[str],
        result_timestamp: Optional[str],
    ) -> LLMAnswerQuality:
        return self.answer_quality_assessor.assess(
            query_text=query_text,
            answer_text=answer_text,
            tool_result_text=tool_result_text,
            knowledge_cutoff=knowledge_cutoff,
            result_timestamp=result_timestamp,
        )

    def _llm_answer_realism(
        self,
        *,
        query_text: str,
        answer_text: str,
        tool_result_text: str,
        knowledge_cutoff: Optional[str],
        result_timestamp: Optional[str],
    ) -> LLMAnswerRealism:
        return self.answer_realism_assessor.assess(
            query_text=query_text,
            answer_text=answer_text,
            tool_result_text=tool_result_text,
            knowledge_cutoff=knowledge_cutoff,
            result_timestamp=result_timestamp,
        )

    def _round_and_clamp(self, v: float) -> float:
        v = _clamp01(v)
        return float(f"{v:.{self.max_decimals}f}")


__all__ = [
    "ConfidenceAssessor",
    "LLMEntityAlignment",
    "LLMAnswerQuality",
    "LLMAnswerRealism",
]
