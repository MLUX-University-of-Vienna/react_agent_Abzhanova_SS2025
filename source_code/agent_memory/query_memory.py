from typing import List, Tuple

class QueryMemory:
    def __init__(self):
        self.query: str = ""
        self.subqueries: List[Tuple[str, str]] = []
