from dataclasses import dataclass
from typing import Dict, Optional, Iterable

@dataclass
class PromptRegistry:
    _defaults: Dict[str, PromptTemplate]
    _overrides: Dict[str, PromptTemplate] = None

    def __post_init__(self):
        if self._overrides is None:
            self._overrides = {}

    def get(self, prompt_id: str) -> PromptTemplate:
        return self._overrides.get(prompt_id) or self._defaults[prompt_id]

    def set(self, prompt_id: str, template: PromptTemplate) -> None:
        if prompt_id not in self._defaults:
            raise KeyError(f"Unknown prompt_id: {prompt_id}")
        self._overrides[prompt_id] = template

    def reset(self, prompt_id: str) -> None:
        self._overrides.pop(prompt_id, None)

    def list_ids(self) -> Iterable[str]:
        return self._defaults.keys()

    def render(self, prompt_id: str, **kwargs) -> str:
        return self.get(prompt_id).render(**kwargs)
