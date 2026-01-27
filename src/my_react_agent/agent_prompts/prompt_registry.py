from dataclasses import dataclass
from typing import Dict, Iterable, Optional

@dataclass
class PromptRegistry:
    _defaults: Dict[str, "PromptTemplate"]
    _overrides: Optional[Dict[str, "PromptTemplate"]] = None

    def __post_init__(self):
        if self._overrides is None:
            self._overrides = {}

    def get(self, prompt_id: str) -> "PromptTemplate":
        if prompt_id in self._overrides:
            return self._overrides[prompt_id]
        if prompt_id in self._defaults:
            return self._defaults[prompt_id]
        raise KeyError(f"Unknown prompt_id: {prompt_id}")

    def set(self, prompt_id: str, template: "PromptTemplate") -> None:
        if prompt_id not in self._defaults:
            raise KeyError(f"Unknown prompt_id: {prompt_id}. Register a default first.")
        self._overrides[prompt_id] = template

    def register_default(self, prompt_id: str, template: "PromptTemplate") -> None:
        if prompt_id in self._defaults:
            return
        self._defaults[prompt_id] = template

    def reset(self, prompt_id: str) -> None:
        self._overrides.pop(prompt_id, None)

    def list_ids(self) -> Iterable[str]:
        return set(self._defaults.keys()) | set(self._overrides.keys())

    def render(self, prompt_id: str, **kwargs) -> str:
        return self.get(prompt_id).render(**kwargs)