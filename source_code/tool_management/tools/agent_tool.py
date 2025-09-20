import re
from abc import ABC, abstractmethod
from typing import Optional, Dict

from agent_core.models import ToolResponse

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
    def execute(self, input: str) -> ToolResponse:
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
