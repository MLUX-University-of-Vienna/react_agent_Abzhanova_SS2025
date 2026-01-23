from dataclasses import dataclass, field
from typing import Dict, Set, Any

@dataclass(frozen=True)
class PromptTemplate:
    text: str
    required_vars: Set[str] = field(default_factory=set)
    description: str = ""
    version: str = "1"

    def render(self, **kwargs: Any) -> str:
        missing = self.required_vars - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing prompt vars: {sorted(missing)}")
        return self.text.format(**kwargs)
