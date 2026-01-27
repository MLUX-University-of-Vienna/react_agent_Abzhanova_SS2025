from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..llm_adapters.llm_base import LLMBase

from .models import ParameterRating
from .parameter_assessor import ParameterAssessor

from ..agent_prompts.prompts_ids import PromptId
from ..agent_prompts.prompt_registry import PromptRegistry

from .json_utils import (
    _safe_text,
    _clamp01,
    _serialise_tool_result,
)

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
        parameter_assessors: Optional[List[ParameterAssessor]] = None,
    ):
        self.llm = llm
        self.default_fallback = _clamp01(default_fallback)
        self.max_decimals = max(0, int(max_decimals))
        self.entity_mismatch_penalty = _clamp01(entity_mismatch_penalty)
        self.include_tool_result_in_quality = bool(include_tool_result_in_quality)
        self.tool_result_max_chars = max(0, int(tool_result_max_chars))
        self.log_parse_failures = bool(log_parse_failures)

        self.parameter_assessors: List[ParameterAssessor] = list(parameter_assessors or [])

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

        ratings = self._run_parameter_assessors(
            query_text=task_text,
            answer_text=summary_text,
            tool_result_text="",
            knowledge_cutoff=knowledge_cutoff,
            result_timestamp=result_timestamp,
        )

        entity = self._pick_entity_rating(ratings)
        quality = self._pick_named_rating(ratings, {"answer_quality", "quality"}, fallback_name="answer_quality")
        realism = self._pick_named_rating(ratings, {"answer_realism", "realism"}, fallback_name="answer_realism")

        entity_same = bool((entity.meta or {}).get("same_entity", True))
        entity_conf = float(entity.score)
        quality_score = float(quality.score)
        realism_score = float(realism.score)

        if not entity_same:
            conf = entity_conf * (1.0 - self.entity_mismatch_penalty)
            out = self._round_and_clamp(conf)
            logger.info(
                "[ConfidenceAssessor.assess_step_summary] entity_same=False entity_conf=%.3f quality=%.3f realism=%.3f "
                "-> conf=%.3f (penalized) reason_entity=%s reason_quality=%s reason_realism=%s",
                entity_conf,
                quality_score,
                realism_score,
                out,
                entity.reason,
                quality.reason,
                realism.reason,
            )
            return out

        conf = self._mean_score(ratings)
        out = self._round_and_clamp(conf)
        logger.info(
            "[ConfidenceAssessor.assess_step_summary] entity_same=True entity_conf=%.3f quality=%.3f realism=%.3f "
            "-> conf=%.3f reason_entity=%s reason_quality=%s reason_realism=%s",
            entity_conf,
            quality_score,
            realism_score,
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

        tool_text_for_assessors = tool_text if self.include_tool_result_in_quality else ""

        ratings = self._run_parameter_assessors(
            query_text=query_text,
            answer_text=answer_text,
            tool_result_text=tool_text_for_assessors,
            knowledge_cutoff=knowledge_cutoff,
            result_timestamp=result_timestamp,
        )

        entity = self._pick_entity_rating(ratings)
        quality = self._pick_named_rating(ratings, {"answer_quality", "quality"}, fallback_name="answer_quality")
        realism = self._pick_named_rating(ratings, {"answer_realism", "realism"}, fallback_name="answer_realism")

        entity_same = bool((entity.meta or {}).get("same_entity", True))
        entity_conf = float(entity.score)
        quality_score = float(quality.score)
        realism_score = float(realism.score)

        if not entity_same:
            conf = entity_conf * (1.0 - self.entity_mismatch_penalty)
            out = self._round_and_clamp(conf)
            logger.info(
                "[ConfidenceAssessor.assess] entity_same=False entity_conf=%.3f quality=%.3f realism=%.3f "
                "-> conf=%.3f (penalized) reason_entity=%s reason_quality=%s reason_realism=%s",
                entity_conf,
                quality_score,
                realism_score,
                out,
                entity.reason,
                quality.reason,
                realism.reason,
            )
            return out

        conf = self._mean_score(ratings)
        out = self._round_and_clamp(conf)
        logger.info(
            "[ConfidenceAssessor.assess] entity_same=True entity_conf=%.3f quality=%.3f realism=%.3f "
            "-> conf=%.3f reason_entity=%s reason_quality=%s reason_realism=%s",
            entity_conf,
            quality_score,
            realism_score,
            out,
            entity.reason,
            quality.reason,
            realism.reason,
        )
        return out

    def _run_parameter_assessors(
        self,
        *,
        query_text: str,
        answer_text: str,
        tool_result_text: str,
        knowledge_cutoff: Optional[str],
        result_timestamp: Optional[str],
    ) -> Dict[str, ParameterRating]:
        out: Dict[str, ParameterRating] = {}

        if not self.parameter_assessors:
            out["fallback"] = ParameterRating(
                name="fallback",
                score=self.default_fallback,
                reason="fallback",
                meta={},
            )
            return out

        for a in self.parameter_assessors:
            try:
                r = a.assess(
                    query_text=query_text,
                    answer_text=answer_text,
                    tool_result_text=tool_result_text,
                    knowledge_cutoff=knowledge_cutoff,
                    result_timestamp=result_timestamp,
                )
            except Exception:
                r = None

            if isinstance(r, ParameterRating) and r.name:
                out[r.name] = r

        if not out:
            out["fallback"] = ParameterRating(
                name="fallback",
                score=self.default_fallback,
                reason="fallback",
                meta={},
            )

        return out

    def _pick_entity_rating(self, ratings: Dict[str, ParameterRating]) -> ParameterRating:
        for r in ratings.values():
            try:
                if isinstance(r.meta, dict) and "same_entity" in r.meta:
                    return r
            except Exception:
                continue

        for key in ("entity_alignment", "entity"):
            if key in ratings:
                return ratings[key]

        return ParameterRating(
            name="entity_alignment",
            score=self.default_fallback,
            reason="fallback",
            meta={"same_entity": True},
        )

    def _pick_named_rating(
        self,
        ratings: Dict[str, ParameterRating],
        names: set,
        *,
        fallback_name: str,
    ) -> ParameterRating:
        for n in names:
            if n in ratings:
                return ratings[n]
        return ParameterRating(name=fallback_name, score=self.default_fallback, reason="fallback", meta={})

    def _mean_score(self, ratings: Dict[str, ParameterRating]) -> float:
        scores: List[float] = []
        for r in ratings.values():
            try:
                if isinstance(r.meta, dict) and r.meta.get("exclude_from_mean", False):
                    continue
            except Exception:
                pass
            scores.append(_clamp01(getattr(r, "score", self.default_fallback)))

        if not scores:
            return float(self.default_fallback)

        return sum(scores) / float(len(scores))

    def _round_and_clamp(self, v: float) -> float:
        v = _clamp01(v)
        return float(f"{v:.{self.max_decimals}f}")


__all__ = [
    "ConfidenceAssessor",
]
