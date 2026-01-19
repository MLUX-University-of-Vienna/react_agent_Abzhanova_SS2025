from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple, Union

from agent_memory.data_structures import (
    Entity,
    EntityId,
    Query,
    QueryId,
    StepId,
    new_entity,
    new_query,
    query_upsert_entity,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


QueryIdLike = Union[QueryId, str]
StepIdLike = Union[StepId, str]
EntityIdLike = Union[EntityId, str]


@dataclass(frozen=True)
class ConversationTurn:
    query_id: QueryId
    step_id: Optional[StepId]
    query_text: str
    answer_text: str
    created_at: datetime


class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns: int = max(1, int(max_turns))

        self.queries: Dict[QueryId, Query] = {}
        self.entities: Dict[EntityId, Entity] = {}

        self._turns_by_query: Dict[QueryId, Deque[ConversationTurn]] = {}

        self.current_query_id: Optional[QueryId] = None

        logger.info("ConversationMemory initialised max_turns=%d", self.max_turns)


    def add_query(self, query: Query, *, set_current: bool = True) -> QueryId:
        self.queries[query.id] = query
        before_ids = set(self.entities.keys())

        ents = list(query.entities or ())
        synced = 0
        for ent in ents:
            eid = getattr(ent, "entity_id", None) or getattr(ent, "id", None)
            if eid is None:
                logger.warning("ConversationMemory: entity missing id field: %r", ent)
                continue
            self.entities[eid] = ent
            synced += 1

        after_ids = set(self.entities.keys())
        new_ids = after_ids - before_ids

        names = []
        for e in ents:
            nm = (getattr(e, "canonical_name", "") or "").strip()
            if nm:
                names.append(nm)
        preview = ", ".join(names[:10]) + (", ..." if len(names) > 10 else "")

        logger.info(
            "ConversationMemory synced query_id=%s entities=%d (new=%d) preview=[%s]",
            str(query.id),
            synced,
            len(new_ids),
            preview,
        )

        if query.id not in self._turns_by_query:
            self._turns_by_query[query.id] = deque(maxlen=self.max_turns)

        if set_current:
            self.current_query_id = query.id
            logger.info("ConversationMemory linked to Query id=%s", str(query.id))

        return query.id

    def create_query(self, original_text: str, *, set_current: bool = True) -> Query:
        q = new_query(original_text)
        self.add_query(q, set_current=set_current)
        return q

    def set_current_query(self, query: Query) -> None:
        self.add_query(query, set_current=True)

    def set_current_query_model(self, query_model: Query) -> None:
        self.set_current_query(query_model)

    def add_entity(self, entity: Entity, *, query_id: Optional[QueryId] = None) -> EntityId:
        self.entities[entity.entity_id] = entity

        qid = query_id or self.current_query_id
        if qid is not None and qid in self.queries:
            q = self.queries[qid]
            q2 = query_upsert_entity(q, entity)
            self.queries[qid] = q2

        return entity.entity_id

    def create_entity(self, canonical_name: str, *, query_id: Optional[QueryId] = None) -> Entity:
        ent = new_entity(canonical_name)
        self.add_entity(ent, query_id=query_id)
        return ent

    def update(
        self,
        query: str,
        answer: str,
        *,
        query_id: Optional[QueryIdLike] = None,
        step_id: Optional[StepIdLike] = None,
    ) -> None:
        qid = self._coerce_query_id(query_id) or self.current_query_id
        sid = self._coerce_step_id(step_id)

        if qid is None:
            q = self.create_query(query, set_current=True)
            qid = q.id
        else:
            self.current_query_id = qid
            if qid not in self.queries:
                self.queries[qid] = Query(id=qid, original_text=query)
                self._turns_by_query[qid] = deque(maxlen=self.max_turns)

        if qid not in self._turns_by_query:
            self._turns_by_query[qid] = deque(maxlen=self.max_turns)

        turn = ConversationTurn(
            query_id=qid,
            step_id=sid,
            query_text=query,
            answer_text=answer,
            created_at=datetime.utcnow(),
        )
        self._turns_by_query[qid].append(turn)

        logger.info(
            "ConversationMemory appended turn query_id=%s q_len=%d a_len=%d",
            str(qid),
            len(query or ""),
            len(answer or ""),
        )

    def get_recent_turns(
        self,
        *,
        query_id: Optional[QueryIdLike] = None,
        limit: Optional[int] = None,
    ) -> Tuple[ConversationTurn, ...]:
        qid = self._coerce_query_id(query_id) or self.current_query_id
        if qid is None:
            return tuple()

        turns = self._turns_by_query.get(qid)
        if turns is None or not turns:
            return tuple()

        n = self.max_turns if limit is None else max(1, int(limit))
        items = list(turns)
        return tuple(items[-n:])

    def get_recent_qa_pairs(
        self,
        *,
        query_id: Optional[QueryIdLike] = None,
        limit: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        turns = self.get_recent_turns(query_id=query_id, limit=limit)
        return [(t.query_text, t.answer_text) for t in turns]

    def get_recent_context(
        self,
        *,
        query_id: Optional[QueryIdLike] = None,
        limit: Optional[int] = None,
    ) -> str:
        pairs = self.get_recent_qa_pairs(query_id=query_id, limit=limit)
        if not pairs:
            return ""
        lines: List[str] = []
        for q, a in pairs:
            lines.append(f"Q: {q}\nA: {a}")
        return "Recent conversation context:\n" + "\n".join(lines)

    def get_relevant_context(self) -> str:
        return self.get_recent_context(query_id=self.current_query_id, limit=self.max_turns)


    def find_entity_by_id(self, entity_id: EntityIdLike) -> Optional[Entity]:
        eid = self._coerce_entity_id(entity_id)
        if eid is None:
            return None
        return self.entities.get(eid)

    def find_entity_by_name(self, name: str) -> Optional[Entity]:
        needle = self._normalise_text(name)
        if not needle:
            return None

        for ent in self.entities.values():
            if self._normalise_text(ent.canonical_name) == needle:
                return ent
            for al in (ent.aliases or ()):
                if self._normalise_text(al) == needle:
                    return ent
        return None


    def _normalise_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _coerce_query_id(self, qid: Optional[QueryIdLike]) -> Optional[QueryId]:
        if qid is None:
            return None
        if isinstance(qid, QueryId):
            return qid
        s = (qid or "").strip()
        return QueryId(s) if s else None

    def _coerce_step_id(self, sid: Optional[StepIdLike]) -> Optional[StepId]:
        if sid is None:
            return None
        if isinstance(sid, StepId):
            return sid
        s = (sid or "").strip()
        return StepId(s) if s else None

    def _coerce_entity_id(self, eid: Optional[EntityIdLike]) -> Optional[EntityId]:
        if eid is None:
            return None
        if isinstance(eid, EntityId):
            return eid
        s = (eid or "").strip()
        return EntityId(s) if s else None
