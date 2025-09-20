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

    @property
    def when_to_pick_head(self) -> Optional[List[str]]:
        return None

    @property
    def examples(self) -> List[str]:
        return []

    @property
    def regex_patterns(self) -> List[str]:
        return []

    def to_catalog_entry(self) -> str:
        lines = [f"- {self.name}"]
        if self.instructions:
            lines.append(f"  instructions: {self.instructions}")
        if self.when_to_pick:
            lines.append(f"  when_to_pick: {self.when_to_pick}")
        head = self.when_to_pick_head
        if head:
            lines.append("  when_to_pick_head:")
            for item in head:
                lines.append(f"    - {item}")
        if self.examples:
            lines.append("  examples:")
            for ex in self.examples:
                lines.append(f"    - {ex}")
        return "\n".join(lines)
