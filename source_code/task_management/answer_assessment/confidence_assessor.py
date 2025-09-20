import logging
import math
import re
from typing import Any, Optional

from llm_adapters.llm_base import LLMBase

logger = logging.getLogger(__name__)

class ConfidenceAssessor:
    def __init__(self, llm: LLMBase, default_fallback: float = 0.5, max_decimals: int = 2):
        self.llm = llm
        self.default_fallback = float(default_fallback)
        self.max_decimals = max(0, int(max_decimals))
        logger.info(f"[ConfidenceAssessor.__init__] Initialised default_fallback={self.default_fallback:.2f} max_decimals={self.max_decimals}")

    def assess(self, query: str, result: Optional[Any], answer: str, knowledge_cutoff: Optional[str] = None, result_timestamp: Optional[str] = None) -> float:
        logger.info("[ConfidenceAssessor.assess] Starting LLM-only confidence assessment")
        query_text = self._safe_to_text(query)
        answer_text = self._safe_to_text(answer)
        result_text = ""
        try:
            if result is None:
                result_text = ""
                logger.debug("[ConfidenceAssessor.assess] Tool result is None; treating as empty text")
            elif hasattr(result, "text") and isinstance(getattr(result, "text"), str):
                result_text = str(getattr(result, "text"))
                logger.debug("[ConfidenceAssessor.assess] Using result.text attribute")
            elif hasattr(result, "content") and isinstance(getattr(result, "content"), str):
                result_text = str(getattr(result, "content"))
                logger.debug("[ConfidenceAssessor.assess] Using result.content attribute")
            else:
                result_text = str(result)
                logger.debug("[ConfidenceAssessor.assess] Using str(result) fallback")
        except Exception as e:
            logger.warning(f"[ConfidenceAssessor.assess] Failed to serialise result error={e!r}; using empty string")
            result_text = ""

        prompt = self._build_prompt(query=query_text, result=result_text, answer=answer_text, knowledge_cutoff=knowledge_cutoff, result_timestamp=result_timestamp)
        logger.debug(f"[ConfidenceAssessor.assess] Prompt length={len(prompt)}")
        try:
            raw = (self.llm.generate(prompt) or "").strip()
            logger.debug(f"[ConfidenceAssessor.assess] Raw LLM output={raw!r}")
            parsed_score = self._parse_score(raw)
            logger.info(f"[ConfidenceAssessor.assess] Parsed confidence score={parsed_score:.3f}")
        except Exception as e:
            logger.warning(f"[ConfidenceAssessor.assess] LLM call failed error={e!r}; using fallback={self.default_fallback:.2f}")
            parsed_score = self._round_and_clamp(self.default_fallback)

        score = self._coerce_float(parsed_score)
        if math.isnan(score) or math.isinf(score):
            logger.warning("[ConfidenceAssessor.assess] Parsed score invalid (nan/inf); coercing to fallback 0.50")
            score = 0.50
        score = max(0.0, min(1.0, score))
        score = float(f"{score:.{self.max_decimals}f}")
        logger.info(f"[ConfidenceAssessor.assess] Final LLM-only confidence={score:.2f}")
        return score

    def _build_prompt(self, query: str, result: str, answer: str, knowledge_cutoff: Optional[str], result_timestamp: Optional[str]) -> str:
        kc = f"Knowledge cutoff: {knowledge_cutoff}\n" if knowledge_cutoff else ""
        ts = f"Tool Result timestamp: {result_timestamp or 'unknown'}\n"
        prompt = (
            "You are a confidence grader.\n\n"
            "Assess how well the Answer satisfies the Question, given the Tool Result (if any).\n"
            "Consider completeness, fidelity to evidence, scope/time alignment, and clarity/precision.\n"
            "Output JSON ONLY as: {\"confidence\": <float 0.0-1.0 with at most two decimals>}.\n\n"
            f"{kc}{ts}"
            "Question:\n"
            f"{query}\n\n"
            "Tool Result (may be empty/noisy):\n"
            f"{result}\n\n"
            "Answer:\n"
            f"{answer}\n"
        )
        logger.debug(f"[ConfidenceAssessor._build_prompt] Built prompt bytes={len(prompt.encode('utf-8'))}")
        return prompt

    def _parse_score(self, raw: str) -> float:
        s = (raw or "").strip()
        m = re.search(r'"confidence"\s*:\s*([01](?:\.\d+)?|\.\d+)', s, re.I)
        if m:
            v = float(m.group(1))
            v2 = self._round_and_clamp(v)
            logger.debug(f"[ConfidenceAssessor._parse_score] Extracted JSON field confidence={v2:.3f}")
            return v2
        m2 = re.findall(r'([01](?:\.\d+)?|\.\d+)', s)
        if m2:
            v = float(m2[-1])
            v2 = self._round_and_clamp(v)
            logger.debug(f"[ConfidenceAssessor._parse_score] Fallback numeric detected={v2:.3f}")
            return v2
        logger.debug("[ConfidenceAssessor._parse_score] Unable to parse confidence; returning default fallback")
        return self._round_and_clamp(self.default_fallback)

    def _safe_to_text(self, obj: Any) -> str:
        try:
            if obj is None:
                return ""
            return re.sub(r"\s{2,}", " ", str(obj).replace("\n", " ").replace("\r", " ").strip())
        except Exception:
            return ""

    def _coerce_float(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            v = float(value)
            return self._round_and_clamp(0.5 if (math.isnan(v) or math.isinf(v)) else v)
        if isinstance(value, str):
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.strip())
            if m:
                try:
                    v = float(m.group(0))
                    return self._round_and_clamp(0.5 if (math.isnan(v) or math.isinf(v)) else v)
                except Exception:
                    logger.debug("[ConfidenceAssessor._coerce_float] Failed to parse float from string; using default 0.8")
                    return self._round_and_clamp(0.8)
        return self._round_and_clamp(0.8)

    def _round_and_clamp(self, v: float) -> float:
        v = max(0.0, min(1.0, float(v)))
        return float(f"{v:.{self.max_decimals}f}")
