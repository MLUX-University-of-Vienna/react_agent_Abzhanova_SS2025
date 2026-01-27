from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, TYPE_CHECKING

from ..agent_memory.data_structures import Step, StepResult, StepToolCall, ToolResponse, StepDecision as MemStepDecision

if TYPE_CHECKING:
    from .action_context import ActionHandlerContext


@dataclass
class ActionHandler(ABC):
    @property
    @abstractmethod
    def action_name(self) -> str: ...

    @abstractmethod
    def run(self, ctx: "ActionHandlerContext") -> Tuple[StepToolCall, StepResult, Step]:
        ...

    def should_assess_result(
        self,
        ctx: "ActionHandlerContext",
        *,
        step: Step,
        decision: "MemStepDecision",
        step_result: StepResult,
    ) -> bool:
        return True



def empty_tool_call(tool: str = "") -> StepToolCall:
    return StepToolCall(
        tool_call_id=type("ToolCallIdProxy", (), {"generate": lambda: None})(),  
        tool=tool,
        tool_input="",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        response=ToolResponse(text=""),
        error=None,
    )
