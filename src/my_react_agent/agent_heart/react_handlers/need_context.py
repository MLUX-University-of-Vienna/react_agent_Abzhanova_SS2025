from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ...agent_memory.data_structures import (
    Evidence,
    EvidenceExtract,
    Step,
    StepMeta,
    StepResult,
    StepToolCall,
    ToolResponse,
    step_add_evidence,
    step_set_result,
)

from ..action_context import ActionHandlerContext
from ..action_handler_base import ActionHandler, empty_tool_call
from ..transcript_formatter import TranscriptFormatter


@dataclass
class NeedContextHandler(ActionHandler):
    @property
    def action_name(self) -> str:
        return "RESOLVE_PRONOUNS_AND_OMITTED ENTITIES"

    def run(self, ctx: ActionHandlerContext) -> Tuple[StepToolCall, StepResult, Step]:
        current_step = ctx.step
        step_index = current_step.plan.index
        original_task = (current_step.plan.instruction or "").strip()

        tool_call = empty_tool_call("resolve_pronouns_and_omitted_entities")

        prev_facts: List[Dict[str, Any]] = []
        if hasattr(ctx.agent, "_collect_all_previous_extracted_facts"):
            prev_facts = ctx.agent._collect_all_previous_extracted_facts() or []

        recent_qa: List[Tuple[str, str]] = []
        if hasattr(ctx.agent, "_get_recent_queries_qa"):
            recent_qa = ctx.agent._get_recent_queries_qa(limit=5) or []

        hit = None
        if hasattr(ctx.agent, "rewrite_step_task_with_context"):
            hit = ctx.agent.rewrite_step_task_with_context(
                step_task=original_task,
                previous_step_facts=prev_facts,
                recent_queries_qa=recent_qa,
            )

        rewritten = original_task
        confidence = 0.4
        extracted = {}
        found = False

        if isinstance(hit, dict):
            rewritten = (hit.get("rewritten_task") or original_task).strip()
            confidence = float(hit.get("confidence", 0.7) or 0.7)
            extracted = hit.get("extracted", {}) or {}
            found = bool(hit.get("found", False))

        formatter = TranscriptFormatter(ctx.format_action_line, ctx.format_observation_line)
        formatter.append_action(ctx.transcript_lines, step_index, "RESOLVE_PRONOUNS_AND_OMITTED ENTITIES", "", original_task)

        obs = (
            f"Rewritten step task: {rewritten}"
            if (found and rewritten and rewritten != original_task)
            else "No rewrite found; keeping the original step task."
        )
        formatter.append_observation(ctx.transcript_lines, step_index, obs)

        evidence = Evidence(
            tool="resolve_pronouns_and_omitted_entities",
            content=obs,
            url=None,
            extracted=EvidenceExtract(fields={"original_task": original_task}, payload={"rewrite": rewritten, "meta": extracted}),
            as_of=datetime.utcnow(),
            confidence=confidence,
        )
        step2 = step_add_evidence(current_step, evidence)

        if found and rewritten and rewritten != original_task:
            try:
                step2 = replace(step2, plan=replace(step2.plan, instruction=rewritten))
            except Exception:
                pass


        step_result = StepResult(
            success=True,
            observation=obs,
            confidence=confidence,
            extracted_entities=tuple(),
            extracted_facts=tuple(),  
            error=None,
            meta=StepMeta(action="RESOLVE_PRONOUNS_AND_OMITTED ENTITIES"),
            should_stop=False,
            final_answer=None,
        )
        step3 = step_set_result(step2, step_result)
        return tool_call, step_result, step3

    def should_assess_result(self, ctx, *, step, decision, step_result) -> bool:
        return False