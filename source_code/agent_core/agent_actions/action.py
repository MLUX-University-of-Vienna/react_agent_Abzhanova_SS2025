from abc import ABC, abstractmethod
from typing import List, Optional

class Action(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
