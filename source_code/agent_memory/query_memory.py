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
        self.subqueries.append((subquery, answer))

    def get_all(self) -> List[Tuple[str, str]]:
        return self.subqueries

    def get_context_for_subquery(self, subquery: str) -> str:
        prior = [f"Q: {q}\nA: {a}" for q, a in self.subqueries if q != subquery]
        return "\n".join(prior[-3:])

    def resolve_pronouns(self, subquery: str) -> str:
        prior_answers = [answer for _, answer in self.subqueries]
        query = self.query
        resolved = self.pronouns_resolver.resolve(
            [query, subquery],
            prior_answers=prior_answers
        )
        return resolved[-1]


