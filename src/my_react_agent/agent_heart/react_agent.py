from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..agent_memory.data_structures import (
    QueryStatus,
    StepPlan,
    Step,
    StepDecision as MemStepDecision,
    Evidence,
    step_set_decision,
    step_set_result,
)

from ..agent_memory.conversation_memory import ConversationMemory
from ..agent_memory.query_memory import QueryMemory
from ..agent_memory.entity_extractor import EntityExtractor

from ..agent_core.agent_actions.action import Action

from ..llm_adapters.llm_base import LLMBase
from ..tool_management.tool_executor import ToolExecutor
from ..tool_management.tool_query_refiner import ToolQueryRefiner
from ..tool_management.tools.agent_tool import AgentTool
from ..confidence_assessment.confidence_assessor import ConfidenceAssessor
from ..confidence_assessment.parameter_assessor import ParameterAssessor

from .logger import logger
from .config import ReActAgentConfig
from .trace_models import ReActStep
from .action_services import ActionServices
from .action_handler_base import ActionHandler
from .transcript_formatter import TranscriptFormatter
from .react_handlers.answer_by_itself import AnswerByItselfHandler
from .react_handlers.stop import StopHandler
from .react_handlers.clarify import ClarifyHandler
from .react_handlers.use_tool import UseToolHandler
from .react_handlers.need_context import NeedContextHandler
from .steps.step_factory import StepFactory
from .steps.step_trace_recorder import StepTraceRecorder
from .action_context import ActionHandlerContext

from .mixins.formatting import FormattingMixin
from .mixins.planning import PlanningMixin
from .mixins.execution import ExecutionMixin
from .mixins.prompting import PromptingMixin


class ReActAgent(
    FormattingMixin,
    PlanningMixin,
    ExecutionMixin,
    PromptingMixin,
):
    def __init__(
        self,
        planner_llm: LLMBase,
        summariser_llm: LLMBase,
        confidence_llm: LLMBase,
        entity_extractor: EntityExtractor,
        refiner_llm: LLMBase,
        tools: Dict[str, AgentTool],
        *,
        max_steps: int = None,
        entity_store: Optional[Any] = None,
        config: Optional[ReActAgentConfig] = None,
        step_actions: Optional[List[Action]] = None,
        low_conf_actions: Optional[List[Action]] = None,
        parameter_assessors: Optional[List[ParameterAssessor]] = None,
        prompts: PromptRegistry,
    ):
        if not tools:
            raise ValueError("ReActAgent requires at least one tool")

        self.planner_llm = planner_llm
        self.summariser_llm = summariser_llm
        self.refiner_llm = refiner_llm
        self.tools = tools

        self.tool_executor = ToolExecutor(tools)

        self.config = config or ReActAgentConfig()
        if max_steps is not None:
            self.config.max_steps = int(max_steps)

        try:
            self.confidence_assessor = ConfidenceAssessor(
                llm=confidence_llm,
                parameter_assessors=parameter_assessors,
            )
        except Exception as e:
            logger.warning(
                "[ReActAgent.__init__] Failed to initialise ConfidenceAssessor error=%r; continuing without tool/step gating",
                e,
            )
            self.confidence_assessor = None

        self.step_actions = step_actions
        self.low_conf_actions = low_conf_actions

        self.entity_extractor = entity_extractor

        self.conversation_memory = ConversationMemory(max_turns=5)
        self.query_memory = QueryMemory()

        self.tool_query_refiner = ToolQueryRefiner(llm=self.refiner_llm, max_chars=300)
        self.entity_store = entity_store

        self.fact_store: Dict[str, Any] = {}

        self._previous_step_results: List[Dict[str, Any]] = []
        self._active_step_dependencies: List[int] = []
        self._step_extracted_facts: Dict[int, List[Dict[str, Any]]] = {}
        self._current_step_index: Optional[int] = None

        self._current_question: str = ""
        self._context_snippets: List[str] = []

        self.prompts = prompts

        logger.info(
            "[ReActAgent.__init__] Initialised with tools=%s max_steps=%d step_actions=%s low_conf_actions=%s",
            list(self.tools.keys()),
            self.config.max_steps,
            [a.name for a in self.step_actions],
            [a.name for a in self.low_conf_actions],
        )

    def _collect_all_previous_extracted_facts(self) -> List[Dict[str, Any]]:
        prev: List[Dict[str, Any]] = getattr(self, "_previous_step_results", []) or []
        out: List[Dict[str, Any]] = []

        for r in reversed(prev):
            if not isinstance(r, dict):
                continue
            step_idx = r.get("index")
            step_instr = r.get("instruction", "")
            ef = r.get("extracted_facts", []) or []
            if isinstance(ef, (list, tuple)):
                for f in ef:
                    if isinstance(f, dict):
                        out.append(
                            {
                                **f,
                                "_step_index": step_idx,
                                "_step_instruction": step_instr,
                            }
                        )
        return out

    def _get_recent_queries_qa(self, *, limit: int = 5) -> List[Tuple[str, str]]:
        cm = getattr(self, "conversation_memory", None)
        if cm is None:
            return []

        turns_by_query = getattr(cm, "_turns_by_query", {}) or {}
        items: List[Tuple[datetime, Any, Any]] = []

        for qid, dq in turns_by_query.items():
            try:
                if not dq:
                    continue
                last_turn = dq[-1]
                ts = getattr(last_turn, "created_at", None)
                if ts is None:
                    ts = datetime.min
                items.append((ts, qid, last_turn))
            except Exception:
                continue

        items.sort(key=lambda x: x[0], reverse=True)

        out: List[Tuple[str, str]] = []
        n = max(1, int(limit or 5))

        for _, qid, last_turn in items[:n]:
            qobj = None
            try:
                qobj = getattr(cm, "queries", {}).get(qid)
            except Exception:
                qobj = None

            original = ""
            final = ""

            if qobj is not None:
                try:
                    original = (getattr(qobj, "original_text", "") or "").strip()
                except Exception:
                    original = ""
                try:
                    final = (getattr(qobj, "final_answer", "") or "").strip()
                except Exception:
                    final = ""

            if not original:
                try:
                    original = (getattr(last_turn, "query_text", "") or "").strip()
                except Exception:
                    original = ""
            if not final:
                try:
                    final = (getattr(last_turn, "answer_text", "") or "").strip()
                except Exception:
                    final = ""

            if original:
                out.append((original, final))

        return out

    def rewrite_step_task_with_context(
        self,
        *,
        step_task: str,
        previous_step_facts: List[Dict[str, Any]],
        recent_queries_qa: List[Tuple[str, str]],
    ) -> Optional[Dict[str, Any]]:
        task = (step_task or "").strip()
        if not task:
            return None

        facts_lines: List[str] = []
        for i, f in enumerate(previous_step_facts or [], start=1):
            if not isinstance(f, dict):
                continue
            v = self._sanitise_to_text(f.get("value") or f.get("text") or "")
            if not v:
                continue
            k = self._sanitise_to_text(f.get("key") or "fact")
            si = f.get("_step_index", None)
            facts_lines.append(f"{i}. [step {si}] {k}: {v}")
            if len(facts_lines) >= 40:
                break
        facts_block = "\n".join(facts_lines) if facts_lines else "(none)"

        queries_lines: List[str] = []
        for i, (q, a) in enumerate(recent_queries_qa or [], start=1):
            q2 = self._sanitise_to_text(q)
            a2 = self._sanitise_to_text(a)
            queries_lines.append(f"{i}. original_text: {q2}\n   final_answer: {a2 or '(empty)'}")
            if len(queries_lines) >= 10:
                break
        queries_block = "\n".join(queries_lines) if queries_lines else "(none)"

        schema = (
            "{"
            '"found": true or false,'
            '"rewritten_task":"<string>",'
            '"confidence": 0.0,'
            '"extracted":{'
            '  "used_sources":["previous_step_facts" or "recent_queries"],'
            '  "resolved_mentions":[{"mention":"<e.g. there>","resolved_to":"<entity>","source":"previous_step_facts|recent_queries"}],'
            '  "notes":"<optional>"'
            "}"
            "}"
        )

        prompt = (
            "You rewrite the CURRENT STEP TASK using only the provided memory-like inputs.\n\n"
            "Goal:\n"
            "- If the task contains pronouns or omitted entities, resolve them using:\n"
            "  1) previous_steps_extracted_facts (newer first)\n"
            "  2) recent_queries (most recent first)\n"
            "- Rewrite the task to be explicit and executable.\n"
            "- Do NOT invent entities. If you cannot resolve ambiguity, set found=false and return rewritten_task unchanged.\n"
            "- Keep the CURRENT STEP TASK the same, just resolve pronouns and omitted entities; do not add new sub-tasks.\n\n"
            f"CURRENT STEP TASK:\n{task}\n\n"
            f"previous_steps_extracted_facts:\n{facts_block}\n\n"
            f"recent_queries (most recent first):\n{queries_block}\n\n"
            f"Return STRICT JSON with this shape:\n{schema}\n\n"
            "JSON only:"
        )

        raw = (self.summariser_llm.generate(prompt) or "").strip()
        data = self._safe_parse_json_object(raw)

        found_raw = data.get("found", False)
        found = bool(found_raw) if isinstance(found_raw, bool) else str(found_raw).strip().lower() in {
            "true",
            "1",
            "yes",
        }

        rewritten = self._sanitise_to_text(data.get("rewritten_task", "")).strip() or task
        if not found:
            rewritten = task

        conf_raw = data.get("confidence", 0.7)
        try:
            confidence = float(conf_raw)
        except Exception:
            confidence = 0.7
        confidence = max(0.0, min(1.0, confidence))

        extracted = data.get("extracted", {}) or {}
        if not isinstance(extracted, dict):
            extracted = {}

        return {
            "found": found,
            "rewritten_task": rewritten,
            "confidence": confidence,
            "extracted": extracted,
        }

    def _finalise_in_memory(self, question: str, final_answer: str) -> None:
        q = self.query_memory.current_query_model
        if q is not None:
            q2 = replace(q, final_answer=final_answer, status=QueryStatus.ANSWERED)
            self.query_memory.current_query_model = q2

            self.conversation_memory.add_query(q2, set_current=True)

            qid = q2.id
        else:
            qid = None

        self.conversation_memory.update(question, final_answer, query_id=qid)

    def _synthesise_final_answer(
        self, question: str, executed_trace: List[ReActStep], context_snippets: List[str]
    ) -> str:
        parts: List[str] = []
        for s in executed_trace:
            parts.append(
                f"- Step {s.index} thought: {s.thought}\n"
                f"  action: {s.action_kind} [{s.action_tool or ''} :: {s.action_input}]\n"
                f"  observation: {s.observation}\n"
                f"  evidence: {s.evidence_snippet or '(none)'}"
            )
        ctx_block = "\n".join(context_snippets[-5:]) if context_snippets else "(none)"

        prompt = (
            "You are producing the final answer to the user's question.\n"
            "You are given a sequence of executed steps (with observations and evidence snippets) and any "
            "additional context snippets.\n\n"
            "Task:\n"
            "- Combine the observations and evidence into a direct, final answer to the user's question.\n"
            "- Aim for 1–3 sentences, unless the question clearly needs more detail.\n"
            "- You may only use numbers or concrete facts that appear in the observations, evidence, or context "
            "  snippets; do NOT invent new numbers and change existing numbers, do NOT round any number even if user asks. \n"
            "- Prepare a direct answer for the user and don't explain what you did unless user explicitly ask.\n"
            "- If the supplied information is insufficient to answer a numeric part of the question, say so explicitly "
            "  instead of guessing.\n\n"
            f"Question:\n{question}\n\n"
            f"Executed steps, observations, and evidence:\n{''.join(parts) if parts else '(none)'}\n\n"
            f"Additional context snippets:\n{ctx_block}\n\n"
            "Final answer:"
        )

        try:
            out = (self.summariser_llm.generate(prompt) or "").strip()
            final_answer = self._sanitise_to_text(out) or "I could not determine the answer conclusively."
        except Exception as e:
            logger.warning("[FinaliseHandler] summariser_llm.generate failed error=%r", e)
            final_answer = "\n".join([p for p in [s.observation for s in executed_trace] if p]).strip() or (
                "I could not determine the answer conclusively."
            )

        return final_answer

    def _reset_run_state(self) -> None:
        self._previous_step_results = []
        self._active_step_dependencies = []
        self._step_extracted_facts = {}
        self._current_step_index = None
        self.fact_store = {}
        self._context_snippets = []

    def _get_handler_for_action(self, action_name: str) -> ActionHandler:
        handler_map: Dict[str, ActionHandler] = {
            "ANSWER_BY_ITSELF": AnswerByItselfHandler(),
            "STOP": StopHandler(),
            "CLARIFY": ClarifyHandler(),
            "RESOLVE_PRONOUNS_AND_OMITTED ENTITIES": NeedContextHandler(),
            "USE_TOOL": UseToolHandler(),
        }
        return handler_map.get(action_name, UseToolHandler())

    def _build_action_services(self) -> ActionServices:
        return ActionServices(
            answer_generator=self.answer_generator,
            tool_executor=self.tool_executor,
            tool_query_refiner=self.tool_query_refiner,
            confidence_assessor=self.confidence_assessor,
        )

    def _pick_step_summary_evidence(self, step: Step) -> Optional[Evidence]:
        if step is None:
            return None
        evs = getattr(step, "evidence", None)
        if not evs or not isinstance(evs, (list, tuple)):
            return None

        for ev in reversed(evs):
            try:
                if getattr(ev, "tool", "") == "step_summary":
                    return ev
            except Exception:
                continue

        try:
            return evs[-1] if evs else None
        except Exception:
            return None

    def _should_retry_low_conf_step(
        self,
        *,
        step_task: str,
        updated_step: Step,
        decision: MemStepDecision,
        action_name: str,
        step_attempt: int,
        max_step_attempts: int,
        should_stop: bool,
        prior_unsuccessful_attempts: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, Optional[MemStepDecision], Optional[float]]:
        if not getattr(self.config, "assess_all_steps", True):
            return False, None, None
        if self.confidence_assessor is None:
            return False, None, None

        threshold = float(getattr(self.config, "step_confidence_threshold", 0.7))

        summary_ev = self._pick_step_summary_evidence(updated_step)
        summary_text = self._sanitise_to_text(getattr(summary_ev, "content", "") or "")

        try:
            conf = self.confidence_assessor.assess_step_summary(
                step_task=self._sanitise_to_text(step_task),
                step_summary_evidence=summary_ev,
                knowledge_cutoff=None,
                result_timestamp=(
                    summary_ev.as_of.isoformat()
                    if (summary_ev is not None and getattr(summary_ev, "as_of", None) is not None)
                    else None
                ),
                context=None,
            )
        except Exception as e:
            logger.exception(
                "[ReActAgent._should_retry_low_conf_step] Step=%d assessment crashed; treating as low confidence",
                updated_step.plan.index,
            )
            conf = 0.0

        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0

        if conf_f >= threshold:
            return False, None, conf_f

        if should_stop:
            return False, None, conf_f

        if step_attempt >= max_step_attempts:
            return False, None, conf_f

        logger.info(
            "[ReActAgent.handle] Step=%d low-confidence (conf=%.2f<th=%.2f); invoking step-level low-conf handler; attempt=%d/%d",
            updated_step.plan.index,
            conf_f,
            threshold,
            step_attempt + 1,
            max_step_attempts,
        )

        history_lines: List[str] = []
        history_lines.append(f"Step {updated_step.plan.index} produced a low-confidence result.")
        history_lines.append(f"Action used: {decision.action}")
        history_lines.append(f"Step task: {self._sanitise_to_text(step_task)}")
        history_lines.append(f"Output: {summary_text}")
        history_lines.append(f"Confidence: {conf_f:.2f} (threshold: {threshold:.2f})")
        history_lines.append(f"Tool name: {getattr(decision, 'tool_name', '') or ''}")
        history_lines.append(f"Tool input: {getattr(decision, 'tool_input', '') or ''}")

        try:
            if prior_unsuccessful_attempts:
                for i, att in enumerate(prior_unsuccessful_attempts, start=1):
                    if not isinstance(att, dict):
                        continue
                    history_lines.append(
                        f"Attempt {i}: action={att.get('action','')}, tool_name={att.get('tool_name','')}, tool_input={att.get('tool_input','')}"
                    )
        except Exception:
            pass

        history_text = "\n".join(history_lines)

        try:
            new_decision = self._handle_low_confidence_step_result(
                question=step_task,
                plan=updated_step.plan,
                decision=decision,
                raw_text=history_text,
                prior_unsuccessful_attempts=prior_unsuccessful_attempts,
            )
            return True, new_decision, conf_f
        except Exception:
            pass

        return False, None, conf_f

    def handle(self, question: str) -> str:
        logger.info("[ReActAgent.handle] started with question length=%d", len(question or ""))
        self._current_question = question or ""

        self.query_memory.start_new(question)
        if self.query_memory.current_query_model:
            self.query_memory.current_query_model = replace(
                self.query_memory.current_query_model, status=QueryStatus.RUNNING
            )
            self.conversation_memory.add_query(self.query_memory.current_query_model, set_current=True)

        self._reset_run_state()

        transcript_lines: List[str] = []
        transcript_lines.append(f"You asked: {self._sanitise_to_text(question)}")

        separate = self._should_separate_question(question)
        logger.info("[ReActAgent.handle] Question separation decision: separate=%s", separate)

        if separate:
            plans = self._plan_question_steps(question)
        else:
            plans = [
                StepPlan(
                    index=1,
                    instruction=self._sanitise_to_text(question),
                    needed_facts=[self._sanitise_to_text(question)],
                    depends_on=[],
                )
            ]
            logger.info("[ReActAgent.handle] Using single-step plan (no separation).")

        try:
            self.query_memory.save_global_plan(plans)
        except Exception:
            pass

        transcript_lines.append("Planned steps:")
        for plan in plans:
            transcript_lines.append(f"- Step {plan.index}: {plan.instruction}")

        executed_trace: List[ReActStep] = []
        context_snippets: List[str] = self._context_snippets

        step_factory = StepFactory()
        trace_recorder = StepTraceRecorder(self)
        services = self._build_action_services()

        for plan_position, plan in enumerate(plans, start=1):
            if plan_position > self.config.max_steps:
                logger.info("[ReActAgent.handle] Max steps reached (%d); stopping execution.", self.config.max_steps)
                break

            self._active_step_dependencies = list(plan.depends_on or [])
            self._current_step_index = plan.index

            decision = self._decide_action_for_step(
                question=question,
                plan=plan,
                executed_plans=plans[: plan_position - 1],
                executed_trace_for_prompt=executed_trace,
            )

            max_step_attempts = (
                int(getattr(self.config, "max_step_retries", 20))
                if getattr(self.config, "assess_all_steps", True)
                else 0
            )
            step_attempt = 0

            need_context_attempt = 0
            need_context_max_attempts_in_one_step = 1

            unsuccessful_step_attempts: List[Dict[str, Any]] = []

            while True:
                action_name = (decision.action or "").upper().strip()

                if (
                    action_name == "RESOLVE_PRONOUNS_AND_OMITTED ENTITIES"
                    and need_context_attempt >= need_context_max_attempts_in_one_step
                ):
                    saved_actions = self.step_actions
                    try:
                        self.step_actions = [
                            a
                            for a in saved_actions
                            if (a.name or "").upper() != "RESOLVE_PRONOUNS_AND_OMITTED ENTITIES"
                        ]
                        decision = self._decide_action_for_step(
                            question=question,
                            plan=plan,
                            executed_plans=plans[: plan_position - 1],
                            executed_trace_for_prompt=executed_trace,
                        )
                    finally:
                        self.step_actions = saved_actions
                    continue

                thought_text = decision.think or f"I will execute step {plan.index}."
                transcript_lines.append(self._format_thought_line(plan.index, thought_text))

                step = step_factory.build_running_step(plan)
                step = step_set_decision(step, decision)

                handler = self._get_handler_for_action(action_name)

                handler_context = ActionHandlerContext(
                    agent=self,
                    step=step,
                    decision=decision,
                    question=question,
                    transcript_lines=transcript_lines,
                    services=services,
                    format_action_line=self._format_action_line_human,
                    format_observation_line=self._format_observation_line,
                )

                tool_call, step_result, updated_step = handler.run(handler_context)

                if action_name == "RESOLVE_PRONOUNS_AND_OMITTED ENTITIES":
                    rewritten_task = (updated_step.plan.instruction or "").strip()
                    if rewritten_task and rewritten_task != plan.instruction:
                        plan = replace(plan, instruction=rewritten_task)
                        plans[plan_position - 1] = plan
                        try:
                            self.query_memory.save_global_plan(plans)
                        except Exception:
                            pass

                    need_context_attempt += 1

                    if need_context_attempt >= need_context_max_attempts_in_one_step:
                        saved_actions = self.step_actions
                        try:
                            self.step_actions = [
                                a
                                for a in saved_actions
                                if (a.name or "").upper() != "RESOLVE_PRONOUNS_AND_OMITTED ENTITIES"
                            ]
                            decision = self._decide_action_for_step(
                                question=question,
                                plan=plan,
                                executed_plans=plans[: plan_position - 1],
                                executed_trace_for_prompt=executed_trace,
                            )
                        finally:
                            self.step_actions = saved_actions
                        continue

                    decision = self._decide_action_for_step(
                        question=question,
                        plan=plan,
                        executed_plans=plans[: plan_position - 1],
                        executed_trace_for_prompt=executed_trace,
                    )
                    continue

                should_stop = bool(getattr(step_result, "should_stop", False))
                final_text = getattr(step_result, "final_answer", None) or step_result.observation

                updated_step = step_factory.finish_step(updated_step)

                try:
                    updated_step, _facts = self._build_step_summary_evidence_and_facts(updated_step)
                except Exception:
                    pass

                assess_this_action = True
                try:
                    assess_this_action = bool(
                        handler.should_assess_result(
                            handler_context,
                            step=updated_step,
                            decision=decision,
                            step_result=step_result,
                        )
                    )
                except Exception:
                    assess_this_action = True

                step_task = self._sanitise_to_text(updated_step.plan.instruction or question)

                if assess_this_action:
                    should_retry, new_decision, conf = self._should_retry_low_conf_step(
                        step_task=step_task,
                        updated_step=updated_step,
                        decision=decision,
                        action_name=action_name,
                        step_attempt=step_attempt,
                        max_step_attempts=max_step_attempts,
                        should_stop=should_stop,
                        prior_unsuccessful_attempts=unsuccessful_step_attempts,
                    )
                else:
                    should_retry, new_decision, conf = False, None, None

                if should_retry and new_decision is not None:
                    try:
                        used_tool_name = getattr(decision, "tool_name", "") or ""
                        used_tool_input = getattr(decision, "tool_input", "") or ""
                        if tool_call is not None:
                            try:
                                if isinstance(tool_call, dict):
                                    used_tool_name = (
                                        tool_call.get("tool_name")
                                        or tool_call.get("name")
                                        or tool_call.get("tool")
                                        or used_tool_name
                                    )
                                    used_tool_input = (
                                        tool_call.get("tool_input")
                                        or tool_call.get("input")
                                        or used_tool_input
                                    )
                                else:
                                    used_tool_name = (
                                        getattr(tool_call, "tool_name", None)
                                        or getattr(tool_call, "name", None)
                                        or getattr(tool_call, "tool", None)
                                        or used_tool_name
                                    )
                                    used_tool_input = (
                                        getattr(tool_call, "tool_input", None)
                                        or getattr(tool_call, "input", None)
                                        or used_tool_input
                                    )
                            except Exception:
                                pass
                        unsuccessful_step_attempts.append(
                            {
                                "action": getattr(decision, "action", "") or "",
                                "tool_name": used_tool_name,
                                "tool_input": used_tool_input,
                            }
                        )
                    except Exception:
                        pass

                    decision = new_decision
                    step_attempt += 1
                    continue

                exhausted = False
                try:
                    threshold = float(getattr(self.config, "step_confidence_threshold", 0.7))
                    if conf is not None and float(conf) < threshold and step_attempt >= max_step_attempts:
                        exhausted = True
                except Exception:
                    exhausted = False

                if exhausted:
                    msg = "Unfortunately,  step result couldn't be defined due to lack of information."
                    try:
                        sr = getattr(updated_step, "result", None)
                        if sr is not None:
                            kwargs: Dict[str, Any] = {}
                            if hasattr(sr, "observation"):
                                kwargs["observation"] = msg
                            if hasattr(sr, "final_answer"):
                                kwargs["final_answer"] = None
                            if hasattr(sr, "should_stop"):
                                kwargs["should_stop"] = False
                            if hasattr(sr, "success"):
                                kwargs["success"] = False
                            if kwargs:
                                sr2 = replace(sr, **kwargs)
                                updated_step = step_set_result(updated_step, sr2)
                    except Exception:
                        pass
                    final_text = msg
                    should_stop = False
                    allow_store = True
                else:
                    allow_store = True
                    if conf is not None:
                        try:
                            threshold = float(getattr(self.config, "step_confidence_threshold", 0.7))
                            if float(conf) < threshold:
                                allow_store = False
                        except Exception:
                            pass

                if allow_store:
                    trace_recorder.update_previous_step_results(updated_step)
                    trace_recorder.persist_step_to_query_memory(step=updated_step)

                    q_now = self.query_memory.current_query_model
                    if q_now is not None:
                        self.conversation_memory.add_query(q_now, set_current=True)

                    try:
                        obs = (updated_step.result.observation if updated_step.result else "") or ""
                        ev_snip = trace_recorder.extract_step_evidence_snippet(updated_step)
                        tool_nm = (updated_step.decision.tool_name if updated_step.decision else "") or None
                        tool_in = (updated_step.decision.tool_input if updated_step.decision else "") or ""
                        if updated_step.result is not None and updated_step.result.success:
                            executed_trace.append(
                                ReActStep(
                                    index=updated_step.plan.index,
                                    thought=thought_text,
                                    action_kind=(
                                        updated_step.decision.action if updated_step.decision else decision.action
                                    ),
                                    action_tool=tool_nm,
                                    action_input=tool_in,
                                    observation=obs,
                                    evidence_snippet=ev_snip or None,
                                )
                            )

                        self._executed_trace = executed_trace
                    except Exception:
                        pass

                try:
                    self.query_memory.set_transcript(transcript_lines)
                except Exception:
                    pass

                if should_stop:
                    final_answer = (
                        final_text
                        or (updated_step.result.observation if updated_step.result else "")
                        or "Stopped before completing all steps."
                    )
                    try:
                        self.query_memory.set_transcript(transcript_lines)
                    except Exception:
                        pass
                    self._finalise_in_memory(question, final_answer)
                    return "\n".join(transcript_lines)

                break

        final_best = self._synthesise_final_answer(question, executed_trace, context_snippets)
        transcript_lines.append(f"Final answer: {final_best}")

        try:
            self.query_memory.set_transcript(transcript_lines)
        except Exception:
            pass

        self._finalise_in_memory(question, final_best)
        return "\n".join(transcript_lines)
