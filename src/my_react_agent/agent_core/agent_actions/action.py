from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Action(ABC):
    def __init__(self) -> None:
        self._instructions_override: Optional[str] = None
        self._when_to_pick_override: Optional[str] = None

    @property
    @abstractmethod
    def name(self) -> str:
        return ""

    @property
    @abstractmethod
    def default_instructions(self) -> str:
        return ""

    @property
    def default_when_to_pick(self) -> str:
        return ""

    @property
    def instructions(self) -> str:
        return self._instructions_override if self._instructions_override is not None else self.default_instructions

    @instructions.setter
    def instructions(self, value: str) -> None:
        self._instructions_override = (value or "").strip()

    @property
    def when_to_pick(self) -> str:
        return self._when_to_pick_override if self._when_to_pick_override is not None else self.default_when_to_pick

    @when_to_pick.setter
    def when_to_pick(self, value: str) -> None:
        self._when_to_pick_override = (value or "").strip()

    def set_instructions(self, value: str) -> None:
        self.instructions = value

    def set_when_to_pick(self, value: str) -> None:
        self.when_to_pick = value

    def clear_overrides(self) -> None:
        self._instructions_override = None
        self._when_to_pick_override = None

    @property
    def when_to_pick_head(self) -> Optional[List[str]]:
        return None

    @property
    def examples(self) -> List[str]:
        return []

    def to_catalog_entry(self) -> str:
        lines = [f"- {self.name}"]
        if self.when_to_pick:
            lines.append(f"  when_to_pick: {self.when_to_pick}")
        if self.instructions:
            lines.append(f"  instructions: {self.instructions}")
        return "\n".join(lines)