from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import re, time, logging

from llm_adapters.llm_base import LLMBase

logger = logging.getLogger(__name__)

@dataclass
class Candidate:
    id: str
    text: str
    source: str
    turn: int
    subquery_index: Optional[int] = None
    type_hint: Optional[str] = None
    gram_number: Optional[str] = None
    numeric_value: Optional[float] = None
    gender: Optional[str] = None
    score: float = 0.0
    query_model_id: Optional[str] = None
    subquery_model_id: Optional[str] = None


class PronounsResolver:
    def __init__(
        self,
        llm: LLMBase,
        max_ctx_chars: int = 4000,
        max_candidates: int = 12,
        llm_weight: float = 1.0,
        entity_store: Optional[object] = None,
    ):
        self.llm = llm
        self.max_ctx_chars = max_ctx_chars
        self.max_candidates = max_candidates
        self.llm_weight = max(0.0, min(1.0, llm_weight))
        self.turn_counter = 0
        self.entity_extractor = getattr(self, "entity_extractor", None)
        self.current_query_model_id: Optional[str] = None
        self.current_subquery_model_id: Optional[str] = None
        self.entity_store = entity_store
        self._pron_re = re.compile(r"\b(he|she|it|they|him|her|them|his|hers|its|their|this|that|these|those)\b", re.IGNORECASE)
        self._math_like_re = re.compile(r"\b(multiply|times|x|divide|over|add|plus|sum|subtract|minus|increase|decrease|percent|ratio|rate)\b", re.IGNORECASE)
        self._num_re = re.compile(r"(?<!\w)(\d[\d,_.]*(?:\.\d+)?)(?!\w)")
        self._clean_num_re = re.compile(r"[,_]")

    def start_new_query(self):
        self.turn_counter += 1
        logger.debug(f"[start_new_query] turn_counter={self.turn_counter}")

    def resolve(
        self,
        actual_query: str,
        subqueries: List[str],
        current_index: int,
        prior_pairs: List[Tuple[str, str]],
        conversation_context: str,
        query_model_id: Optional[str] = None,
        subquery_model_id: Optional[str] = None,
    ) -> str:
        if query_model_id:
            self.current_query_model_id = query_model_id
        if subquery_model_id:
            self.current_subquery_model_id = subquery_model_id

        t0 = time.time()
        target = subqueries[current_index]
        if not target or not self._pron_re.search(target):
            return target

        resolved = target
        prons = [m.group(0) for m in self._pron_re.finditer(target)]

        if self._math_like_re.search(target):
            num = self._best_numeric_from_store_or_context(prior_pairs, conversation_context, actual_query, target, query_model_id, subquery_model_id)
            if num is not None:
                for p in prons:
                    if p.lower() in {"it", "this", "that", "its"}:
                        resolved = self._replace_token(resolved, p, num)
                self._log_finish(target, resolved, t0)
                return resolved

        head_text = self._best_entity_text_with_store(prior_pairs, actual_query, conversation_context, target, query_model_id, subquery_model_id)
        if head_text:
            for p in prons:
                if p.lower() in {"they", "them", "their", "these", "those"}:
                    resolved = self._replace_token(resolved, p, head_text)

        if resolved == target:
            last_text = self._fallback_text(prior_pairs, actual_query)
            if last_text:
                for p in prons:
                    if p.lower() in {"it", "this", "that", "its"}:
                        resolved = self._replace_token(resolved, p, last_text)

        self._log_finish(target, resolved, t0)
        return resolved

    def _best_numeric_from_store_or_context(self, pairs: List[Tuple[str, str]], conversation_context: str, actual_query: str, target_subquery: str, query_model_id: Optional[str], subquery_model_id: Optional[str]) -> Optional[str]:
        entity_name = self._pick_entity_name(actual_query, pairs, conversation_context, target_subquery, query_model_id, subquery_model_id)
        predicate_hint = self._detect_predicate(target_subquery) or self._detect_predicate(actual_query)
        if self.entity_store and entity_name:
            v = self._first_numeric_fact_from_store(entity_name, predicate_hint)
            if v is not None:
                return v
        for q, a in reversed(pairs):
            v = self._first_number(a) or self._first_number(q)
            if v is not None:
                return v
        v = self._first_number(conversation_context) or self._first_number(actual_query)
        return v

    def _first_number(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        m = self._num_re.search(text)
        if not m:
            return None
        raw = m.group(1)
        cleaned = self._clean_num_re.sub("", raw)
        return cleaned

    def _best_entity_text_with_store(self, pairs: List[Tuple[str, str]], actual_query: str, conversation_context: str, target_subquery: str, query_model_id: Optional[str], subquery_model_id: Optional[str]) -> Optional[str]:
        ex = self.entity_extractor if self.entity_extractor else None
        if ex:
            e0 = ex.extract(target_subquery, query_id=query_model_id or "", subquery_id=subquery_model_id)
            if e0 and getattr(e0, "text", None):
                return e0.text
        for q, a in reversed(pairs):
            if ex:
                e = ex.extract(q, query_id=query_model_id or "", subquery_id=subquery_model_id) or ex.extract(a, query_id=query_model_id or "", subquery_id=subquery_model_id)
                if e and getattr(e, "text", None):
                    return e.text
        if ex:
            e2 = ex.extract(actual_query, query_id=query_model_id or "", subquery_id=subquery_model_id) or ex.extract(conversation_context, query_id=query_model_id or "", subquery_id=subquery_model_id)
            if e2 and getattr(e2, "text", None):
                return e2.text
        return None

    def _fallback_text(self, pairs: List[Tuple[str, str]], actual_query: str) -> Optional[str]:
        for q, _ in reversed(pairs):
            m = re.search(r"\b(?:of|about|for)\s+([^,?.;:]+)", q, flags=re.IGNORECASE)
            if m:
                t = m.group(1).strip()
                if t:
                    return t
        m2 = re.search(r"\b(?:what|which)\s+(?:is|are)\s+([^,?.;:]+)", actual_query, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).strip()
        return None

    def _replace_token(self, text: str, token: str, replacement: str) -> str:
        pat = re.compile(rf"\b{re.escape(token)}\b", flags=re.IGNORECASE)
        return pat.sub(replacement, text)

    def _log_finish(self, before: str, after: str, t0: float) -> None:
        elapsed_ms = (time.time() - t0) * 1000.0
        logger.info(f"[resolve] changed={after!=before} before={before} after={after} elapsed_ms={elapsed_ms:.1f}")

    def _pick_entity_name(self, actual_query: str, pairs: List[Tuple[str, str]], conversation_context: str, target_subquery: str, query_model_id: Optional[str], subquery_model_id: Optional[str]) -> Optional[str]:
        ex = self.entity_extractor if self.entity_extractor else None
        order = [target_subquery, actual_query, conversation_context] + [q for q, _ in reversed(pairs)] + [a for _, a in reversed(pairs)]
        if ex:
            for t in order:
                if not t:
                    continue
                e = ex.extract(t, query_id=query_model_id or "", subquery_id=subquery_model_id)
                if e and getattr(e, "text", None):
                    return e.text
        return None

    def _detect_predicate(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        t = text.lower()
        keys = [
            "population","price","revenue","gdp","area","cost","length","distance","size","age","height","weight","capacity","count","amount","total","rate","percentage","percent"
        ]
        for k in keys:
            if re.search(rf"\b{k}\b", t):
                return k
        m = re.search(r"\bof\s+([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*)", text)
        if m:
            return None
        return None

    def _first_numeric_fact_from_store(self, entity_name: str, predicate_hint: Optional[str]) -> Optional[str]:
        if not self.entity_store:
            return None
        ent = self.entity_store.resolve(entity_name) if hasattr(self.entity_store, "resolve") else None
        if not ent:
            ent = self.entity_store.ensure_entity(entity_name) if hasattr(self.entity_store, "ensure_entity") else None
        if not ent:
            return None
        eid = getattr(ent, "entity_id", None) or getattr(ent, "id", None)
        if not eid:
            return None
        if predicate_hint and hasattr(self.entity_store, "get_best_fact"):
            f = self.entity_store.get_best_fact(eid, predicate_hint)
            v = getattr(f, "value", None) if f else None
            if self._is_number_like(v):
                return self._format_number_output(v)
        if hasattr(self.entity_store, "all_facts"):
            facts = self.entity_store.all_facts(eid)
            for f in reversed(facts):
                v = getattr(f, "value", None)
                if self._is_number_like(v):
                    return self._format_number_output(v)
        return None

    def _is_number_like(self, v: Any) -> bool:
        if isinstance(v, (int, float)):
            return True
        if isinstance(v, str):
            try:
                float(v.replace(",", ""))
                return True
            except Exception:
                return False
        return False

    def _format_number_output(self, v: Any) -> str:
        try:
            x = float(v)
        except Exception:
            s = str(v)
            return s
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s
