from __future__ import annotations
from typing import Any

from ..agent_prompts.prompts_ids import PromptId
from ..agent_prompts.prompt_registry import PromptRegistry
from ..logger import logger


class PromptingMixin:
    prompts: PromptRegistry

    def _render_prompt(self, pid: PromptId, **vars: Any) -> str:
        try:
            return self.prompts.render(pid, **vars)
        except Exception as e:
            logger.exception("[PromptingMixin] Failed to render prompt %s: %r", pid, e)
            raise