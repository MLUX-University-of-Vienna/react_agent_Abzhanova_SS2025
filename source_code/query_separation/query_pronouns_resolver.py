import re
from typing import List

class QueryPronounsResolver:
    def resolve(self, subqueries: List[str]) -> List[str]:
        if not subqueries:
            return subqueries

        