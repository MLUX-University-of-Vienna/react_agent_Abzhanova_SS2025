from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent_memory.data_structures import Step, StepDecision as MemStepDecision

from .action_services import ActionServices


@dataclass
class ActionHandlerContext:
    step: Step
    decision: MemStepDecision
    question: str
    transcript_lines: List[str]

    services: ActionServices
    agent: Any

    format_action_line: Optional[Callable[..., str]] = None
    format_observation_line: Optional[Callable[..., str]] = None

    extra: Dict[str, Any] = field(default_factory=dict)
