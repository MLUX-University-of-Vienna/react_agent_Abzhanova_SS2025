from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

from ...agent_memory.data_structures import (
    Evidence,
    EvidenceExtract,
    Step,
    StepMeta,
    StepResult,
    StepToolCall,
    step_add_evidence,
    step_set_result,
)

from ..action_context import ActionHandlerContext
from ..action_handler_base import ActionHandler, empty_tool_call
from ..transcript_formatter import TranscriptFormatter

@dataclass
class StopHandler(ActionHandler):
    @property
    def action_name(self) -> str:
        return "STOP"

    def run(self, ctx: ActionHandlerContext) -> Tuple[StepToolCall, StepResult, Step]:
        current_step = ctx.step
        step_index = current_step.plan.index

        tool_call = empty_tool_call("stop")
        message_text = (ctx.decision.message or "").strip() or "Stopping."

        formatter = TranscriptFormatter(ctx.format_action_line, ctx.format_observation_line)
        formatter.append_action(ctx.transcript_lines, step_index, "STOP", "", "", message=message_text)
        formatter.append_observation(ctx.transcript_lines, step_index, message_text)

        evidence = Evidence(
            tool="stop",
            content=message_text,
            url=None,
            extracted=EvidenceExtract(fields={}, payload=None),
            as_of=datetime.utcnow(),
            confidence=1.0,
        )
        step_with_evidence = step_add_evidence(current_step, evidence)

        step_result = StepResult(
            success=False,
            observation=message_text,
            confidence=0.0,
            extracted_entities=tuple(),
            extracted_facts=tuple(), 
            error="stopped",
            meta=StepMeta(action="STOP"),
            should_stop=True,
            final_answer=message_text,
        )
        final_step = step_set_result(step_with_evidence, step_result)
        return tool_call, step_result, final_step

    def should_assess_result(self, ctx, *, step, decision, step_result) -> bool:
        return False