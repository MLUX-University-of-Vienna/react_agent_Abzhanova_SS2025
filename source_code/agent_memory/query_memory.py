from typing import List, Tuple
from agent_memory.pronouns_resolution.pronouns_resolver import PronounsResolver

class QueryMemory:
    def __init__(self, pronouns_resolver: PronounsResolver):
        self.query: str = ""
        self.subqueries: List[Dict[str, Optional[Entity]]] = []
        self.pronouns_resolver = pronouns_resolver

    def start_new(self, query: str):
        self.query = query
        self.subqueries = []

    def add_subquery(self, subquery: str, answer: str):
        q_ent: Optional[Entity] = self.pronouns_resolver.entity_extractor.extract(subquery)
        a_ent: Optional[Entity] = self.pronouns_resolver.entity_extractor.extract(answer)
        self.subqueries.append({
            "subquery":   subquery,
            "entity":     q_ent,
            "answer":     answer,
            "answer_ent": a_ent,
        })

    def get_all(self) -> List[Tuple[str, str]]:
        return self.subqueries

    def get_context_for_subquery(self, subquery: str) -> str:
        prior = [
            f"Q: {r['subquery']}\nA: {r['answer']}"
            for r in self.subqueries
            if r["subquery"] != subquery
        ]
        return "\n".join(prior[-3:])

    def resolve_pronouns(self, subquery: str) -> str:
        prior_answers = [answer for _, answer in self.subqueries]
        query = self.query
        resolved = self.pronouns_resolver.resolve(
            [query, subquery],
            prior_answers=prior_answers
        )
        return resolved[-1]


