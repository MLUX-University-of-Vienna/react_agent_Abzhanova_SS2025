from abc import ABC, abstractmethod
from typing import Optional, List

from agent_memory.entity import Entity


class EntityExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> Optional[Entity]:
        pass

    def extract_all(self, text: str) -> List[Entity]:
        one = self.extract(text)
        return [one] if one else []
