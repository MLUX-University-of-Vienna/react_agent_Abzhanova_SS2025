from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Union

from .prompt_template import PromptTemplate
from .prompts_ids import PromptId

PromptKey = Union[str, PromptId]

@dataclass
class PromptRegistry:
    _defaults: Dict[str, PromptTemplate]
    _overrides: Optional[Dict[str, PromptTemplate]] = None

    def __post_init__(self):
        if self._overrides is None:
            self._overrides = {}

    def _key(self, prompt_id: PromptKey) -> str:
        return prompt_id.value if isinstance(prompt_id, PromptId) else str(prompt_id)

    def get(self, prompt_id: PromptKey) -> PromptTemplate:
        k = self._key(prompt_id)
        return self._overrides.get(k) or self._defaults[k]

    def set(self, prompt_id: PromptKey, template: PromptTemplate) -> None:
        k = self._key(prompt_id)
        if k not in self._defaults:
            raise KeyError(f"Unknown prompt_id: {k}")
        self._overrides[k] = template

    def reset(self, prompt_id: PromptKey) -> None:
        k = self._key(prompt_id)
        self._overrides.pop(k, None)

    def list_ids(self) -> Iterable[str]:
        return self._defaults.keys()

    def render(self, prompt_id: PromptKey, **kwargs) -> str:
        return self.get(prompt_id).render(**kwargs)