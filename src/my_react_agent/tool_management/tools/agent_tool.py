from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ...agent_memory.data_structures import Evidence


class AgentTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute(self, input: str) -> Evidence:
        raise NotImplementedError

    @classmethod
    def get_examples(cls) -> str:
        return ""

    @property
    def refiner_instructions(self) -> str:
        return ""

    @property
    def refiner_input_format(self) -> str:
        return ""

    @property
    def refiner_input_regex(self) -> Optional[str]:
        return None

    @property
    def refiner_forbidden(self) -> str:
        return ""

    @property
    def refiner_examples(self) -> str:
        return ""

    @property
    def refiner_max_chars(self) -> int:
        return 1000
