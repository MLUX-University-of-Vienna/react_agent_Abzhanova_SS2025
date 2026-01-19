from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Tuple

from agent_memory.data_structures import (
    Evidence,
    EvidenceExtract,
    Step,
    StepMeta,
    StepResult,
    StepToolCall,
    ToolResponse,
    step_add_evidence,
    step_add_tool_call,
    step_set_result,
)

from ..action_context import ActionHandlerContext
from ..action_handler_base import ActionHandler
from ..transcript_formatter import TranscriptFormatter


@dataclass
class UseToolHandler(ActionHandler):
    @property
    def action_name(self) -> str:
        return "USE_TOOL"

    def run(self, ctx: ActionHandlerContext) -> Tuple[StepToolCall, StepResult, Step]:
        current_step = ctx.step
        step_plan = current_step.plan
        step_decision = ctx.decision

        tool_name = (step_decision.tool_name or "").strip()
        if not tool_name:
            tool_registry = getattr(ctx.agent, "tools", {}) or {}
            tool_name = next(iter(tool_registry.keys()), "")

        raw_tool_input = (step_plan.instruction or "").strip()

        started_at = datetime.utcnow()

        tool_call = StepToolCall(
            tool_call_id=type("ToolCallIdProxy", (), {"generate": lambda: None})(),  
            tool=tool_name,
            tool_input=raw_tool_input,
            started_at=started_at,
            finished_at=None, 
            response=None,
            error=None,
        )

        formatter = TranscriptFormatter(ctx.format_action_line, ctx.format_observation_line)
        formatter.append_action(ctx.transcript_lines, step_plan.index, "USE_TOOL", tool_name, raw_tool_input)

        observation_text, evidence_snippet, raw_text, extracted_facts = ctx.agent._execute_and_observe(
            tool_name=tool_name,
            tool_input=raw_tool_input,
            user_question=ctx.question,
            subquestion=step_plan.instruction,
        )

        tool_call = replace(
            tool_call,
            finished_at=datetime.utcnow(),
            response=ToolResponse(text=(raw_text or observation_text or ""), raw=None),
            error=None,
        )

        evidence = Evidence(
            tool=tool_name,
            content=(raw_text or observation_text or ""),
            url=None,
            extracted=EvidenceExtract(fields={}, payload=None),
            as_of=datetime.utcnow(),
            confidence=0.8,
        )

        step_with_tool_call = step_add_tool_call(current_step, tool_call)
        step_with_evidence = step_add_evidence(step_with_tool_call, evidence)

        formatter.append_observation(ctx.transcript_lines, step_plan.index, observation_text or raw_text)

        success = bool((observation_text or "").strip())

        step_result = StepResult(
            success=success,
            observation=(observation_text or raw_text),
            confidence=0.8,
            extracted_entities=tuple(),
            extracted_facts=tuple(extracted_facts) if isinstance(extracted_facts, list) else tuple(),
            error=None if success else "tool_low_confidence_or_empty",
            meta=StepMeta(action="USE_TOOL", tool=tool_name, refined_input=raw_tool_input),
            should_stop=False,
            final_answer=None,
        )

        final_step = step_set_result(step_with_evidence, step_result)
        return tool_call, step_result, final_step

