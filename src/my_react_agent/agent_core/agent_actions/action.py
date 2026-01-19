from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Action(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        return ""

    @property
    @abstractmethod
    def instructions(self) -> str:
        return ""

    @property
    def when_to_pick(self) -> str:
        return ""

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

