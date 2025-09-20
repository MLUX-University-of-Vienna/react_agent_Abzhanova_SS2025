# agent_memory/conversation_memory.py
from typing import List, Dict, Optional
import logging
import re
from uuid import uuid4
from agent_memory.pronouns_resolution.pronouns_resolver import PronounsResolver
from agent_memory.pronouns_resolution.entity_extractor import EntityExtractor
from .entity import Mention, Fact
from agent_core.models import Query as QueryModel, Subquery as SubqueryModel
from agent_core.enums import QueryStatus, SubqueryStatus, ExecutionMode

logger = logging.getLogger(__name__)

class ConversationMemory:
    def __init__(self, pronouns_resolver: PronounsResolver):
        self.history: List[Dict[str, object]] = []
        self.pronouns_resolver = pronouns_resolver
        self.MAX_TURNS = 5
        self.current_query_model: Optional[QueryModel] = None
        self._alias_pairs: List[Dict[str, str]] = []

    def set_current_query_model(self, query_model: QueryModel):
        self.current_query_model = query_model

    def update(self, query: str, answer: str, subquery_id: Optional[str] = None, facts: Optional[List[Fact]] = None, query_id: Optional[str] = None):
        extractor: Optional[EntityExtractor] = getattr(self.pronouns_resolver, "entity_extractor", None)
        query_model_id = (query_id or getattr(self.current_query_model, "id", "") or "").strip() or None
        q_ent = extractor.extract(query, query_id=query_model_id or "", subquery_id=subquery_id) if extractor else None
        a_ent = extractor.extract(answer, query_id=query_model_id or "", subquery_id=subquery_id) if extractor else None
        self.history.append({
            "query": query,
            "query_entity": q_ent,
            "answer": answer,
            "answer_entity": a_ent,
            "query_model_id": query_model_id,
            "subquery_id": subquery_id,
            "facts": list(facts) if facts else None,
        })
        if q_ent and a_ent:
            qn = self._normalize_text(q_ent.text)
            an = self._normalize_text(a_ent.text)
            if qn and an and qn != an:
                self._record_alias(q_ent.text, a_ent.text)
        logger.debug(f"Updated conversation memory. Current length: {len(self.history)}")

    def attach_facts_to_last_turn(self, facts: List[Fact], where: str = "answer"):
        if not self.history:
            return
        turn = self.history[-1]
        key = "answer_facts" if where == "answer" else "query_facts"
        existing = turn.get(key) or []
        turn[key] = list(existing) + list(facts or [])

    def get_relevant_context(self) -> str:
        if not self.history:
            return ""
        recent_turns = self.history[-self.MAX_TURNS:]
        context_lines: List[str] = []
        for turn in recent_turns:
            question_text = str(turn.get("query") or "")
            answer_text = str(turn.get("answer") or "")
            context_lines.append(f"Q: {question_text}\nA: {answer_text}")
        return "Recent conversation context:\n" + "\n".join(context_lines)

    def resolve_pronouns(self, text: str) -> str:
        prior_pairs = [(str(h.get("query") or ""), str(h.get("answer") or "")) for h in self.history[-self.MAX_TURNS:]]
        actual_query = str(self.history[-1].get("query")) if self.history else text
        conv_ctx = self.get_relevant_context()
        return self.pronouns_resolver.resolve(
            actual_query=actual_query,
            subqueries=[text],
            current_index=0,
            prior_pairs=prior_pairs,
            conversation_context=conv_ctx,
        )

    def resolve_alias(self, phrase: str) -> Optional[str]:
        key = self._normalize_text(phrase)
        for pair in reversed(self._alias_pairs):
            if self._normalize_text(pair.get("source", "")) == key:
                return pair.get("target")
        return None

    def _record_alias(self, source_phrase: str, canonical_entity: str) -> None:
        s = (source_phrase or "").strip()
        t = (canonical_entity or "").strip()
        if not s or not t:
            return
        norm_existing = {(self._normalize_text(p.get("source", "")), self._normalize_text(p.get("target", ""))) for p in self._alias_pairs}
        cand = (self._normalize_text(s), self._normalize_text(t))
        if cand not in norm_existing:
            self._alias_pairs.append({"source": s, "target": t})

    def _normalize_text(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())
