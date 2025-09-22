from abc import ABC, abstractmethod
from typing import List, Optional

class Action(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def instructions(self) -> str: ...

    @property
    def when_to_pick(self) -> str:
        return ""

