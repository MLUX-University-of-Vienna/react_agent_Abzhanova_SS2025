from abc import ABC, abstractmethod
from typing import Optional

class EntityExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> Optional[str]:
        pass