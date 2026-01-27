from __future__ import annotations
from typing import Any

from ..agent_prompts.prompts_ids import PromptId
from ..agent_prompts.prompt_registry import PromptRegistry

class PromptedComponent:
    def __init__(self, *, prompts: PromptRegistry):
        self.prompts = prompts

    def _render_prompt(self, pid: PromptId, **vars: Any) -> str:
        return self.prompts.render(pid.value, **vars)