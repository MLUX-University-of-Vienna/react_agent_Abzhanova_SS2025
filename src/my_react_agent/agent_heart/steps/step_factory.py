from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from agent_memory.data_structures import Step, StepPlan, StepStatus, new_step

@dataclass(frozen=True)
class StepFactory:
    def build_running_step(self, plan: StepPlan) -> Step:
        s = new_step(plan)
        return replace(
            s,
            status=StepStatus.RUNNING,
            decision=None,
            tool_calls=tuple(),
            evidence=tuple(),
            result=None,
            started_at=datetime.utcnow(),
            finished_at=None,
        )

    def finish_step(self, step: Step) -> Step:
        status_out = StepStatus.DONE
        if step.result is not None and not step.result.success:
            status_out = StepStatus.FAILED
        return replace(step, status=status_out, finished_at=datetime.utcnow())
