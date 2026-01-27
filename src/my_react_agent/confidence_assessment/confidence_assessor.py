from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Iterable, Tuple

from ..llm_adapters.llm_base import LLMBase

from .models import ParameterRating
from .parameter_assessor import ParameterAssessor

from ..agent_prompts.prompt_registry import PromptRegistry

try:
    from ..agent_prompts.prompt_template import PromptTemplate
except Exception: 
    PromptTemplate = Any  

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
        prompts: Optional[PromptRegistry] = None,
        default_fallback: float = 0.5,
        max_decimals: int = 2,
        entity_mismatch_penalty: float = 0.98,
        include_tool_result_in_quality: bool = True,
        tool_result_max_chars: int = 12000,
        log_parse_failures: bool = True,
        parameter_assessors: Optional[List[ParameterAssessor]] = None,
    ):
        self.llm = llm
        self.prompts = prompts 

        self.default_fallback = _clamp01(default_fallback)
        self.max_decimals = max(0, int(max_decimals))
        self.entity_mismatch_penalty = _clamp01(entity_mismatch_penalty)
        self.include_tool_result_in_quality = bool(include_tool_result_in_quality)
        self.tool_result_max_chars = max(0, int(tool_result_max_chars))
        self.log_parse_fail_failures = bool(log_parse_failures)

        self.parameter_assessors: List[ParameterAssessor] = []
        for a in (parameter_assessors or []):
            self.add_parameter_assessor(a)

        logger.info(
            "[ConfidenceAssessor.__init__] fallback=%.2f max_decimals=%d mismatch_penalty=%.2f "
            "include_tool_result_in_quality=%s tool_result_max_chars=%d log_parse_failures=%s assessors=%d",
            self.default_fallback,
            self.max_decimals,
            self.entity_mismatch_penalty,
            self.include_tool_result_in_quality,
            self.tool_result_max_chars,
            self.log_parse_fail_failures,
            len(self.parameter_assessors),
        )

    def add_parameter_assessor(self, assessor: ParameterAssessor) -> None:
        """Register a new assessor at runtime (framework/plugin use-case)."""
        if assessor is None:
            return
        self._bind_prompts_to_assessor(assessor)
        self._register_assessor_prompt_defaults(assessor)
        self.parameter_assessors.append(assessor)

    def _bind_prompts_to_assessor(self, assessor: ParameterAssessor) -> None:
        """Duck-typed binding so assessors can optionally use PromptRegistry."""
        if self.prompts is None:
            return

        fn = getattr(assessor, "set_prompts", None)
        if callable(fn):
            try:
                fn(self.prompts)
                return
            except Exception as e:
                logger.warning("[ConfidenceAssessor] assessor.set_prompts failed assessor=%r err=%r", assessor, e)

        if hasattr(assessor, "prompts"):
            try:
                current = getattr(assessor, "prompts", None)
                if current is None:
                    setattr(assessor, "prompts", self.prompts)
            except Exception as e:
                logger.warning("[ConfidenceAssessor] failed to set assessor.prompts assessor=%r err=%r", assessor, e)

    def _register_assessor_prompt_defaults(self, assessor: ParameterAssessor) -> None:
        """
        If the assessor ships prompt templates, register them into PromptRegistry.
        This allows plugins to bring their own prompts without editing core defaults.
        """
        if self.prompts is None:
            return

        reg_default = getattr(self.prompts, "register_default", None)
        if not callable(reg_default):
            return

        get_many = getattr(assessor, "default_prompt_templates", None)
        if callable(get_many):
            try:
                items = get_many()
                if isinstance(items, dict):
                    for pid, tpl in items.items():
                        if pid and tpl is not None:
                            reg_default(str(pid), tpl)
                return
            except Exception as e:
                logger.warning("[ConfidenceAssessor] default_prompt_templates failed assessor=%r err=%r", assessor, e)

        pid = getattr(assessor, "prompt_id", None)
        get_one = getattr(assessor, "default_prompt_template", None)
        if pid and callable(get_one):
            try:
                tpl = get_one()
                if tpl is not None:
                    reg_default(str(pid), tpl)
            except Exception as e:
                logger.warning("[ConfidenceAssessor] default_prompt_template failed assessor=%r err=%r", assessor, e)

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

        if not entity_same:
            conf = entity_conf * (1.0 - self.entity_mismatch_penalty)
            return self._round_and_clamp(conf)

        return self._round_and_clamp(self._mean_score(ratings))

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
        entity_same = bool((entity.meta or {}).get("same_entity", True))
        entity_conf = float(entity.score)

        if not entity_same:
            conf = entity_conf * (1.0 - self.entity_mismatch_penalty)
            return self._round_and_clamp(conf)

        return self._round_and_clamp(self._mean_score(ratings))

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