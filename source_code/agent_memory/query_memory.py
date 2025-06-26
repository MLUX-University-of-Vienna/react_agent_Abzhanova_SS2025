from typing import List, Tuple

class QueryMemory:
    def __init__(self):
        self.query: str = ""
        self.subqueries: List[Tuple[str, str]] = []

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
        resolved = self.pronoun_resolver.resolve(["", subquery])
        return resolved[-1]

