from typing import List, Tuple

class QueryMemory:
    def __init__(self):
        self.query: str = ""
        self.subqueries: List[Tuple[str, str]] = []

    def start_new(self, query: str):
        self.query = query
        self.subqueries = []
