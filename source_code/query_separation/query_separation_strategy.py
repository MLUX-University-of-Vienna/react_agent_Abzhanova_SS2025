from abc import ABC, abstractmethod
from typing import List

class QuerySeparationStrategy(ABC):
    @abstractmethod
    def separate(self, query: str) -> List[str]:
        """Break a single user query into a list of subqueries."""
        pass
