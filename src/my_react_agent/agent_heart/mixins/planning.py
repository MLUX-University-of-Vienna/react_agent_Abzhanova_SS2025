from __future__ import annotations

from typing import Dict, List, Optional

from agent_memory.data_structures import StepPlan, StepDecision as MemStepDecision
from agent_core.agent_actions.action import Action

from ..logger import logger
from ..trace_models import ReActStep
from .text_utils import TextUtilsMixin


class PlanningMixin(TextUtilsMixin):
    def _tool_catalog_text(self) -> str:
        lines: List[str] = []
        for name, tool in self.tools.items():
            try:
                desc = (getattr(tool, "description", "") or "").strip().replace("\n", " ")
                examples_block = ""
                if hasattr(tool, "get_examples"):
                    try:
                        examples_block = tool.get_examples() or ""
                    except Exception:
                        examples_block = ""
                lines.append(f"- {name}\n  description: {desc}\n")
                ex_lines = [e.strip() for e in (examples_block or "").splitlines() if e.strip()]
                if ex_lines:
                    lines.append("  examples:")
                    for e in ex_lines[:5]:
                        lines.append(f"    - {e}")
            except Exception as e:
                lines.append(f"- {name}\n  description: (error building catalogue entry: {e})")
        return "\n".join(lines) if lines else "(none)"

    def _should_separate_question(self, question: str) -> bool:
        schema = '{"separate": true or false, "reason": "<short>"}'

        prompt = (
            "You are deciding whether a user question should be decomposed "
            "into multiple sequential steps, or handled as a single step. \n\n"
            "Guidelines: \n"
            "- Answer true in 'separate' if the question clearly involves: \n"
            "  multiple entities to compare \n"
            "  multiple calculations or time periods \n"
            "  dependent sub-questions (where one answer is needed for another) \n"
            "  complex reasoning that benefits from ordered steps \n"
            "- Answer false in 'separate' if the question is a single, direct lookup, "
            "  a simple factual question, or a brief conceptual explanation. \n"
            "- Be conservative: prefer false if you are unsure. \n\n"
            f"User question:\n{question} \n\n"
            f"Return STRICT JSON with this shape:\n{schema} \n\n"
            "JSON only, no explanation:"
        )

        logger.info("[ReActAgent._should_separate_question] Prompt length=%d", len(prompt))
        logger.debug("[ReActAgent._should_separate_question] Prompt:\n%s", prompt)

        raw = (self.planner_llm.generate(prompt) or "").strip()
        logger.info("[ReActAgent._should_separate_question] RAW: %s", raw or "")

        data = self._safe_parse_json_object(raw)
        sep_raw = data.get("separate", False)

        separate = False
        if isinstance(sep_raw, bool):
            separate = sep_raw
        elif isinstance(sep_raw, str):
            v = sep_raw.strip().lower()
            if v in {"true", "yes", "y"}:
                separate = True
            elif v in {"false", "no", "n"}:
                separate = False

        reason = self._sanitise_to_text(data.get("reason", ""))
        logger.info("[ReActAgent._should_separate_question] separate=%s reason=%s", separate, reason)
        return separate

    def _plan_question_steps(self, question: str) -> List[StepPlan]:
        tool_catalog = self._tool_catalog_text()
        schema = (
            '{"steps":['
            '{"index":1,'
            '"instruction":"<short imperative>",'
            '"needed_facts":["<fact>", "..."],'
            '"depends_on":[1,2]},'
            '...]}'
        )

        prompt = (
            "Your task is to break these question into a small, ordered list "
            "of concrete steps that, if executed in sequence, will provide all "
            "facts or/and actions needed to answer the question.\n"
            "Think of these like you're solving a math problem in a minimal numbers of steps. \n\n"
            "Rules:\n"
            "- Steps must be in executable order (1,2,3,...).\n"
            "- Each step must be small, focused on one micro-task. \n"
            "- All steps must use same words from question, no synonyms, no similar words, no resolving of definitions. \n"
            "- If a step answering needs results of previous steps and they contain the same entity, list their indices "
            "  in 'depends_on'. Otherwise, use an empty list.\n"
            "- Do not write step if it will resolve the same fact as step before. \n"
            "- Only describe *what* to do, not *how*, not with which *tool*. \n"
            "- Do not add steps to find information that is already given. \n"
            "- Use at most 6 steps unless the question clearly needs more.\n\n"
            f"User question:\n{question}\n\n"
            f"Available tools (for later execution):\n{tool_catalog}\n\n"
            f"Return STRICT JSON with the following shape:\n{schema}\n\n"
            "JSON only, no explanation:"
        )

        logger.info("[ReActAgent._plan_question_steps] Prompt length=%d", len(prompt))
        logger.debug("[ReActAgent._plan_question_steps] Prompt:\n%s", prompt)

        raw = (self.planner_llm.generate(prompt) or "").strip()
        logger.info("[ReActAgent._plan_question_steps] RAW: %s", raw or "")

        data = self._safe_parse_json_object(raw)
        steps_raw = data.get("steps", [])

        planned_steps: List[StepPlan] = []
        if isinstance(steps_raw, list):
            for default_index, item in enumerate(steps_raw, start=1):
                if not isinstance(item, dict):
                    continue
                idx = item.get("index", default_index)
                try:
                    idx = int(idx)
                except Exception:
                    idx = default_index
                instruction_text = self._sanitise_to_text(item.get("instruction") or item.get("description") or "")
                if not instruction_text:
                    instruction_text = self._sanitise_to_text(question)

                needed_facts_raw = item.get("needed_facts", [])
                if not isinstance(needed_facts_raw, list):
                    needed_facts_raw = []
                needed_facts = [self._sanitise_to_text(x) for x in needed_facts_raw if self._sanitise_to_text(x)]

                depends_on_raw = item.get("depends_on", [])
                depends_on: List[int] = []
                if isinstance(depends_on_raw, list):
                    for dependency_index in depends_on_raw:
                        try:
                            depends_on.append(int(dependency_index))
                        except Exception:
                            continue

                planned_steps.append(
                    StepPlan(index=idx, instruction=instruction_text, needed_facts=needed_facts, depends_on=depends_on)
                )

        if not planned_steps:
            planned_steps.append(
                StepPlan(
                    index=1,
                    instruction=self._sanitise_to_text(question),
                    needed_facts=[self._sanitise_to_text(question)],
                    depends_on=[],
                )
            )

        planned_steps.sort(key=lambda s: s.index)
        logger.info("[ReActAgent._plan_question_steps] Planned %d steps", len(planned_steps))
        for planned_step in planned_steps:
            logger.info(
                "[ReActAgent._plan_question_steps] Step %d: %s | needed_facts=%s | depends_on=%s",
                planned_step.index,
                planned_step.instruction,
                planned_step.needed_facts,
                planned_step.depends_on,
            )
        return planned_steps

    def _decide_action_for_step(
        self,
        question: str,
        plan: StepPlan,
        executed_plans: List[StepPlan],
        executed_trace_for_prompt: Optional[List[ReActStep]] = None,
    ) -> MemStepDecision:
        tool_catalog = self._tool_catalog_text()
        actions_catalog = self._actions_catalog_text(self.step_actions)
        allowed_names = [a.name for a in self.step_actions]
        allowed_names_str = ", ".join(allowed_names)

        previous_step_lines: List[str] = []
        execution_trace = executed_trace_for_prompt or []
        trace_by_index: Dict[int, ReActStep] = {t.index: t for t in execution_trace}

        for prior_plan in executed_plans[-self.config.max_recent_steps_in_prompt :]:
            trace_item = trace_by_index.get(prior_plan.index)
            previous_step_lines.append(
                f"Step {prior_plan.index}: {prior_plan.instruction}\n"
                f"  action: {(trace_item.action_kind if trace_item else '(not executed yet)')}\n"
                f"  observation: {(trace_item.observation if trace_item else '(none)')}"
            )
        prev_block = "\n\n".join(previous_step_lines) if previous_step_lines else "(none)"

        schema = (
            "{"
            '"think":"<short>",'
            f'"action":"one of: {allowed_names_str}",'
            '"tool_name":"<tool name or empty>",'
            '"message":"<optional message>'
            "}"
        )

        prompt = (
            "You are deciding how to execute task of the ONE planned step inside a larger plan. \n\n"
            "Available actions (choose exactly ONE), read precisely:\n"
            f"{actions_catalog}\n\n"
            "Tools you can call, precisely read tool descriptions, tool choice is very important:\n"
            f"{tool_catalog}\n\n"
            f"User question:\n{question}\n\n"
            f"Current step:\n"
            f"- task to execute: {plan.instruction}\n"
            f"- needed_facts: {plan.needed_facts}\n\n"
            f"Previously executed steps:\n{prev_block}\n\n"
            f"Return STRICT JSON matching this shape:\n{schema}\n"
            "JSON:"
        )

        raw = (self.planner_llm.generate(prompt) or "").strip()
        logger.info("[ReActAgent._decide_action_for_step] Step=%d RAW: %s", plan.index, raw or "")
        logger.info("[ReActAgent._decide_action_for_step] Plan instruction=%s", plan.instruction or "")

        data = self._safe_parse_json_object(raw)
        think = self._sanitise_to_text(data.get("think", ""))
        action = self._sanitise_to_text(data.get("action", "")).upper()
        tool_name = self._sanitise_to_text(data.get("tool_name", ""))

        step_task_raw = data.get("step's_task", "")
        step_task = self._sanitise_to_text(step_task_raw)
        tool_input = self._sanitise_to_text(data.get("tool_input", ""))
        if step_task:
            tool_input = step_task

        # FIX: if tool_input is empty, default to the step instruction (step task)
        if not (tool_input or "").strip():
            tool_input = self._sanitise_to_text(plan.instruction or "")

        message = self._sanitise_to_text(data.get("message", ""))

        allowed_set = set(a.upper() for a in allowed_names)
        if action not in allowed_set:
            fallback = "USE_TOOL" if "USE_TOOL" in allowed_set else next(iter(allowed_set))
            logger.info(
                "[ReActAgent._decide_action_for_step] Invalid action '%s'; falling back to '%s'",
                action,
                fallback,
            )
            action = fallback

        logger.info(
            "[ReActAgent._decide_action_for_step] Step=%d Parsed action=%s tool=%s tool_input_len=%d",
            plan.index,
            action,
            tool_name,
            len(tool_input or ""),
        )

        return MemStepDecision(
            action=action,
            think=think,
            tool_name=tool_name,
            tool_input=tool_input,
            message=message,
        )
