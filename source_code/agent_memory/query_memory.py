# agent_memory/query_memory.py
from typing import List, Dict, Tuple, Optional, Any
from uuid import uuid4

from agent_memory.pronouns_resolution.pronouns_resolver import PronounsResolver
from agent_memory.pronouns_resolution.entity_extractor import EntityExtractor
from agent_memory.entity import Mention

from agent_core.models import Query as QueryModel, Subquery as SubqueryModel
from agent_core.enums import QueryStatus, SubqueryStatus, ExecutionMode

class QueryMemory:
    def __init__(self, pronouns_resolver: PronounsResolver):
        self.query: str = ""
        self.subqueries: List[Dict[str, Optional[Mention]]] = []
        self.pronouns_resolver = pronouns_resolver
        self.current_query_model: Optional[QueryModel] = None

    def start_new(self, query: str):
        self.query = query
        self.subqueries = []
        self.current_query_model = QueryModel(id=str(uuid4()), original_text=query, status=QueryStatus.PLANNED)

    def add_subquery(self, subquery: str, answer: str):
        extractor: Optional[EntityExtractor] = getattr(self.pronouns_resolver, "entity_extractor", None)

        subquery_id: Optional[str] = None
        if self.current_query_model:
            existing_model: Optional[SubqueryModel] = None
            for s in self.current_query_model.subqueries:
                if s.text == subquery:
                    existing_model = s
                    break
            if existing_model:
                subquery_id = existing_model.id
            else:
                subquery_id = str(uuid4())

        q_ent = extractor.extract(subquery, query_id=(self.current_query_model.id if self.current_query_model else ""), subquery_id=subquery_id) if extractor else None
        a_ent = extractor.extract(answer,   query_id=(self.current_query_model.id if self.current_query_model else ""), subquery_id=subquery_id) if extractor else None

        self.subqueries.append({
            "subquery":   subquery,
            "entity":     q_ent,
            "answer":     answer,
            "answer_ent": a_ent,
        })

        if self.current_query_model:
            existing = None
            for s in self.current_query_model.subqueries:
                if s.text == subquery:
                    existing = s
                    break
            if existing:
                existing.answer = answer
                if existing.status in {SubqueryStatus.PLANNED, SubqueryStatus.RUNNING}:
                    existing.status = SubqueryStatus.ANSWERED
            else:
                m = SubqueryModel(
                    id=subquery_id or str(uuid4()),
                    text=subquery,
                    mode=ExecutionMode.SELF_ANSWER,
                    status=SubqueryStatus.ANSWERED,
                    answer=answer,
                )
                self.current_query_model.subqueries.append(m)

    def get_all(self) -> List[Dict[str, object]]:
        return self.subqueries

    def get_context_for_subquery(self, subquery: str) -> str:
        prior = [
            f"Q: {r['subquery']}\nA: {r['answer']}"
            for r in self.subqueries
            if r["subquery"] != subquery
        ]
        return "\n".join(prior[-3:])

    def resolve_pronouns(self, subquery: str) -> str:
        prior_pairs = [(r["subquery"], r["answer"]) for r in self.subqueries]
        return self.pronouns_resolver.resolve(
            actual_query=self.query,
            subqueries=[subquery],
            current_index=0,
            prior_pairs=prior_pairs,
            conversation_context="",
        )
