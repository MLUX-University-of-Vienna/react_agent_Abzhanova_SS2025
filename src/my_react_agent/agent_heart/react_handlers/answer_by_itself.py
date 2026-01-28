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
class AnswerByItselfHandler(ActionHandler):
    @property
    def action_name(self) -> str:
        return "ANSWER_BY_ITSELF"

    def run(self, ctx: ActionHandlerContext) -> Tuple[StepToolCall, StepResult, Step]:
        current_step = ctx.step
        step_index = current_step.plan.index

        tool_call = empty_tool_call("answer_by_itself")

        answer_text = (ctx.agent._answer_by_itself_for_step(current_step) or "").strip()

        formatter = TranscriptFormatter(ctx.format_action_line, ctx.format_observation_line)
        formatter.append_action(ctx.transcript_lines, step_index, "ANSWER_BY_ITSELF", "", current_step.plan.instruction)
        formatter.append_observation(ctx.transcript_lines, step_index, answer_text)

        evidence = Evidence(
            tool="answer_by_itself",
            content=answer_text,
            url=None,
            extracted=EvidenceExtract(fields={}, payload=None),
            as_of=datetime.utcnow(),
            confidence=0.6,
        )
        step_with_evidence = step_add_evidence(current_step, evidence)

        step_result = StepResult(
            success=True,
            observation=answer_text,
            confidence=0.6,
            extracted_entities=tuple(),
            extracted_facts=tuple(), 
            error=None,
            meta=StepMeta(action="ANSWER_BY_ITSELF"),
            should_stop=False,
            final_answer=None,
        )
        final_step = step_set_result(step_with_evidence, step_result)
        return tool_call, step_result, final_step

    def should_assess_result(self, ctx, *, step, decision, step_result) -> bool:
        return True