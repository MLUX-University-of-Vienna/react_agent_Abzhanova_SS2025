from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

from ..agent_prompts.prompt_registry import PromptRegistry
from ..agent_prompts.prompts_ids import PromptId

from .logger import logger
from .config import ReActAgentConfig
from .trace_models import ReActStep
from .action_services import ActionServices
from .action_handler_base import ActionHandler
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

ParameterAssessorFactory = Callable[[LLMBase, PromptRegistry], ParameterAssessor]
ParameterAssessorLike = Union[ParameterAssessor, ParameterAssessorFactory]


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
        prompts: PromptRegistry,
        max_steps: int = None,
        entity_store: Optional[Any] = None,
        config: Optional[ReActAgentConfig] = None,
        step_actions: Optional[List[Action]] = None,
        low_conf_actions: Optional[List[Action]] = None,
        parameter_assessors: Optional[List[ParameterAssessorLike]] = None,
    ):
        if not tools:
            raise ValueError("ReActAgent requires at least one tool")

        self.prompts = prompts
        logger.info(
            "[ReActAgent.__init__] PromptRegistry ready defaults=%d overrides=%d",
            len(getattr(self.prompts, "_defaults", {}) or {}),
            len(getattr(self.prompts, "_overrides", {}) or {}),
        )

        self.planner_llm = planner_llm
        self.summariser_llm = summariser_llm
        self.refiner_llm = refiner_llm
        self.tools = dict(tools)

        self.tool_executor = ToolExecutor(self.tools)

        self.config = config or ReActAgentConfig()
        if max_steps is not None:
            self.config.max_steps = int(max_steps)

        self.step_actions = list(step_actions or [])
        self.low_conf_actions = list(low_conf_actions or [])

        self.entity_extractor = entity_extractor

        self.conversation_memory = ConversationMemory(max_turns=5)
        self.query_memory = QueryMemory()

        self.tool_query_refiner = ToolQueryRefiner(llm=self.refiner_llm, prompts=self.prompts, max_chars=300)

        self.entity_store = entity_store
        self.fact_store: Dict[str, Any] = {}

        self._previous_step_results: List[Dict[str, Any]] = []
        self._active_step_dependencies: List[int] = []
        self._step_extracted_facts: Dict[int, List[Dict[str, Any]]] = {}
        self._current_step_index: Optional[int] = None
        self._current_question: str = ""
        self._context_snippets: List[str] = []

        built_assessors: List[ParameterAssessor] = []
        for item in (parameter_assessors or []):
            try:
                if callable(item):
                    a = item(confidence_llm, self.prompts)
                    built_assessors.append(a)
                    logger.info(
                        "[ReActAgent.__init__] Built ParameterAssessor from factory: %s",
                        type(a).__name__,
                    )
                else:
                    built_assessors.append(item)
                    logger.info(
                        "[ReActAgent.__init__] Using provided ParameterAssessor instance: %s",
                        type(item).__name__,
                    )
            except Exception as e:
                logger.exception("[ReActAgent.__init__] Failed building assessor from %r err=%r", item, e)

        try:
            self.confidence_assessor = ConfidenceAssessor(
                llm=confidence_llm,
                prompts=self.prompts,
                parameter_assessors=built_assessors,
            )
            logger.info(
                "[ReActAgent.__init__] ConfidenceAssessor enabled assessors=%d",
                len(getattr(self.confidence_assessor, "parameter_assessors", []) or []),
            )
        except Exception as e:
            logger.warning(
                "[ReActAgent.__init__] ConfidenceAssessor init failed err=%r; continuing without step gating",
                e,
            )
            self.confidence_assessor = None

        logger.info(
            "[ReActAgent.__init__] Ready tools=%d max_steps=%d step_actions=%d low_conf_actions=%d",
            len(self.tools),
            self.config.max_steps,
            len(self.step_actions),
            len(self.low_conf_actions),
        )
        logger.debug(
            "[ReActAgent.__init__] Tools=%s | StepActions=%s | LowConfActions=%s",
            list(self.tools.keys()),
            [getattr(a, "name", type(a).__name__) for a in self.step_actions],
            [getattr(a, "name", type(a).__name__) for a in self.low_conf_actions],
        )

    def register_tool(self, name: str, tool: AgentTool) -> None:
        if not name or tool is None:
            return
        self.tools[name] = tool
        self.tool_executor = ToolExecutor(self.tools)
        logger.info("[ReActAgent.register_tool] Registered tool=%s total_tools=%d", name, len(self.tools))

    def register_action(self, action: Action, *, low_conf: bool = False) -> None:
        if action is None:
            return
        target = self.low_conf_actions if low_conf else self.step_actions
        target.append(action)
        logger.info(
            "[ReActAgent.register_action] Registered action=%s low_conf=%s total_step=%d total_low=%d",
            getattr(action, "name", type(action).__name__),
            low_conf,
            len(self.step_actions),
            len(self.low_conf_actions),
        )

    def register_parameter_assessor(self, assessor: ParameterAssessor) -> None:
        if assessor is None:
            return
        if self.confidence_assessor is None:
            logger.warning(
                "[ReActAgent.register_parameter_assessor] ConfidenceAssessor disabled; cannot register %s",
                type(assessor).__name__,
            )
            return
        self.confidence_assessor.add_parameter_assessor(assessor)
        logger.info(
            "[ReActAgent.register_parameter_assessor] Registered assessor=%s total=%d",
            type(assessor).__name__,
            len(getattr(self.confidence_assessor, "parameter_assessors", []) or []),
        )

    # ---------------------------
    # Context helpers
    # ---------------------------
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
                        out.append({**f, "_step_index": step_idx, "_step_instruction": step_instr})
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
                ts = getattr(last_turn, "created_at", None) or datetime.min
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

        prompt = self._render_prompt(
            PromptId.REWRITE_STEP_TASK_WITH_CONTEXT,
            task=task,
            facts_block=facts_block,
            queries_block=queries_block,
            schema=schema,
        )
        logger.debug(
            "[ReActAgent.rewrite_step_task_with_context] prompt_len=%d facts=%d recent_q=%d",
            len(prompt),
            len(previous_step_facts or []),
            len(recent_queries_qa or []),
        )

        raw = (self.summariser_llm.generate(prompt) or "").strip()
        data = self._safe_parse_json_object(raw)

        found_raw = data.get("found", False)
        found = bool(found_raw) if isinstance(found_raw, bool) else str(found_raw).strip().lower() in {"true", "1", "yes"}

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

        logger.info(
            "[ReActAgent.rewrite_step_task_with_context] found=%s confidence=%.2f task_changed=%s",
            found,
            confidence,
            (rewritten != task),
        )

        return {"found": found, "rewritten_task": rewritten, "confidence": confidence, "extracted": extracted}

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

    def _synthesise_final_answer(self, question: str, executed_trace: List[ReActStep], context_snippets: List[str]) -> str:
        parts: List[str] = []
        for s in executed_trace:
            parts.append(
                f"- Step {s.index} thought: {s.thought}\n"
                f"  action: {s.action_kind} [{s.action_tool or ''} :: {s.action_input}]\n"
                f"  observation: {s.observation}\n"
                f"  evidence: {s.evidence_snippet or '(none)'}"
            )
        steps_block = "\n".join(parts) if parts else "(none)"
        ctx_block = "\n".join((context_snippets or [])[-5:]) if context_snippets else "(none)"

        prompt = self._render_prompt(
            PromptId.SYNTHESISE_FINAL_ANSWER,
            question=question,
            steps_block=steps_block,
            ctx_block=ctx_block,
        )
        logger.debug(
            "[ReActAgent._synthesise_final_answer] prompt_len=%d steps=%d ctx_snips=%d",
            len(prompt),
            len(executed_trace or []),
            len(context_snippets or []),
        )

        try:
            out = (self.summariser_llm.generate(prompt) or "").strip()
            final_answer = self._sanitise_to_text(out) or "I could not determine the answer conclusively."
        except Exception as e:
            logger.warning("[ReActAgent._synthesise_final_answer] summariser_llm.generate failed err=%r", e)
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
        except Exception:
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
            logger.info(
                "[ReActAgent._should_retry_low_conf_step] Step=%d conf=%.2f >= threshold=%.2f -> ok",
                updated_step.plan.index,
                conf_f,
                threshold,
            )
            return False, None, conf_f

        if should_stop or step_attempt >= max_step_attempts:
            logger.info(
                "[ReActAgent._should_retry_low_conf_step] Step=%d conf=%.2f < threshold=%.2f but stop=%s attempts=%d/%d -> no retry",
                updated_step.plan.index,
                conf_f,
                threshold,
                should_stop,
                step_attempt,
                max_step_attempts,
            )
            return False, None, conf_f

        logger.info(
            "[ReActAgent._should_retry_low_conf_step] Step=%d low-confidence conf=%.2f<th=%.2f attempt=%d/%d",
            updated_step.plan.index,
            conf_f,
            threshold,
            step_attempt + 1,
            max_step_attempts,
        )

        summary_text = self._sanitise_to_text(getattr(summary_ev, "content", "") or "")
        history_lines: List[str] = [
            f"Step {updated_step.plan.index} produced a low-confidence result.",
            f"Action used: {decision.action}",
            f"Step task: {self._sanitise_to_text(step_task)}",
            f"Output: {summary_text}",
            f"Confidence: {conf_f:.2f} (threshold: {threshold:.2f})",
            f"Tool name: {getattr(decision, 'tool_name', '') or ''}",
            f"Tool input: {getattr(decision, 'tool_input', '') or ''}",
        ]

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
        except Exception as e:
            logger.warning("[ReActAgent._should_retry_low_conf_step] low-conf handler failed err=%r", e)

        return False, None, conf_f

    def handle(self, question: str) -> str:
        logger.info("[ReActAgent.handle] start question_len=%d", len(question or ""))
        self._current_question = question or ""

        self.query_memory.start_new(question)
        if self.query_memory.current_query_model:
            self.query_memory.current_query_model = replace(self.query_memory.current_query_model, status=QueryStatus.RUNNING)
            self.conversation_memory.add_query(self.query_memory.current_query_model, set_current=True)

        self._reset_run_state()

        transcript_lines: List[str] = [f"You asked: {self._sanitise_to_text(question)}"]

        separate = self._should_separate_question(question)
        logger.info("[ReActAgent.handle] separate_question=%s", separate)

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
            logger.info("[ReActAgent.handle] using single-step plan")

        try:
            self.query_memory.save_global_plan(plans)
        except Exception:
            logger.debug("[ReActAgent.handle] query_memory.save_global_plan failed (non-fatal)")

        transcript_lines.append("Planned steps:")
        for plan in plans:
            transcript_lines.append(f"- Step {plan.index}: {plan.instruction}")

        executed_trace: List[ReActStep] = []
        context_snippets: List[str] = self._context_snippets

        step_factory = StepFactory()
        trace_recorder = StepTraceRecorder(self)
        services = self._build_action_services()

        logger.info("[ReActAgent.handle] executing steps=%d max_steps=%d", len(plans), self.config.max_steps)

        for plan_position, plan in enumerate(plans, start=1):
            if plan_position > self.config.max_steps:
                logger.info("[ReActAgent.handle] max_steps reached (%d); stopping", self.config.max_steps)
                break

            self._active_step_dependencies = list(plan.depends_on or [])
            self._current_step_index = plan.index

            logger.info(
                "[ReActAgent.handle] Step=%d/%d start idx=%d deps=%s",
                plan_position,
                len(plans),
                plan.index,
                self._active_step_dependencies or "[]",
            )

            decision = self._decide_action_for_step(
                question=question,
                plan=plan,
                executed_plans=plans[: plan_position - 1],
                executed_trace_for_prompt=executed_trace,
            )
            logger.info(
                "[ReActAgent.handle] Step=%d initial decision action=%s tool=%s",
                plan.index,
                (decision.action or "").upper().strip(),
                getattr(decision, "tool_name", "") or "",
            )

            max_step_attempts = int(getattr(self.config, "max_step_retries", 20)) if getattr(self.config, "assess_all_steps", True) else 0
            step_attempt = 0

            need_context_attempt = 0
            need_context_max_attempts_in_one_step = 1

            unsuccessful_step_attempts: List[Dict[str, Any]] = []

            while True:
                action_name = (decision.action or "").upper().strip()

                if action_name == "RESOLVE_PRONOUNS_AND_OMITTED ENTITIES" and need_context_attempt >= need_context_max_attempts_in_one_step:
                    logger.info("[ReActAgent.handle] Step=%d need-context max attempts reached; re-deciding without that action", plan.index)
                    saved_actions = self.step_actions
                    try:
                        self.step_actions = [a for a in saved_actions if (a.name or "").upper() != "RESOLVE_PRONOUNS_AND_OMITTED ENTITIES"]
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
                logger.debug(
                    "[ReActAgent.handle] Step=%d handler=%s action=%s tool_call=%s",
                    plan.index,
                    type(handler).__name__,
                    action_name,
                    bool(tool_call),
                )

                if action_name == "RESOLVE_PRONOUNS_AND_OMITTED ENTITIES":
                    rewritten_task = (updated_step.plan.instruction or "").strip()
                    if rewritten_task and rewritten_task != plan.instruction:
                        plan = replace(plan, instruction=rewritten_task)
                        plans[plan_position - 1] = plan
                        logger.info("[ReActAgent.handle] Step=%d plan instruction rewritten", plan.index)
                        try:
                            self.query_memory.save_global_plan(plans)
                        except Exception:
                            logger.debug("[ReActAgent.handle] save_global_plan failed after rewrite (non-fatal)")

                    need_context_attempt += 1

                    if need_context_attempt >= need_context_max_attempts_in_one_step:
                        logger.info("[ReActAgent.handle] Step=%d switching off need-context action for re-decision", plan.index)
                        saved_actions = self.step_actions
                        try:
                            self.step_actions = [a for a in saved_actions if (a.name or "").upper() != "RESOLVE_PRONOUNS_AND_OMITTED ENTITIES"]
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
                    logger.debug("[ReActAgent.handle] Step=%d build_step_summary failed (non-fatal)", plan.index)

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
                                    used_tool_name = tool_call.get("tool_name") or tool_call.get("name") or tool_call.get("tool") or used_tool_name
                                    used_tool_input = tool_call.get("tool_input") or tool_call.get("input") or used_tool_input
                                else:
                                    used_tool_name = getattr(tool_call, "tool_name", None) or getattr(tool_call, "name", None) or getattr(tool_call, "tool", None) or used_tool_name
                                    used_tool_input = getattr(tool_call, "tool_input", None) or getattr(tool_call, "input", None) or used_tool_input
                            except Exception:
                                pass
                        unsuccessful_step_attempts.append(
                            {"action": getattr(decision, "action", "") or "", "tool_name": used_tool_name, "tool_input": used_tool_input}
                        )
                    except Exception:
                        pass

                    decision = new_decision
                    step_attempt += 1
                    logger.info(
                        "[ReActAgent.handle] Step=%d retrying attempt=%d/%d new_action=%s new_tool=%s",
                        plan.index,
                        step_attempt,
                        max_step_attempts,
                        (decision.action or "").upper().strip(),
                        getattr(decision, "tool_name", "") or "",
                    )
                    continue

                exhausted = False
                try:
                    threshold = float(getattr(self.config, "step_confidence_threshold", 0.7))
                    if conf is not None and float(conf) < threshold and step_attempt >= max_step_attempts:
                        exhausted = True
                except Exception:
                    exhausted = False

                if exhausted:
                    msg = "Unfortunately, step result couldn't be defined due to lack of information."
                    logger.info("[ReActAgent.handle] Step=%d exhausted retries -> storing failure message", plan.index)
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
                                updated_step = step_set_result(updated_step, replace(sr, **kwargs))
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
                                logger.info(
                                    "[ReActAgent.handle] Step=%d conf=%.2f<th=%.2f -> not storing as evidence/facts",
                                    plan.index,
                                    float(conf),
                                    threshold,
                                )
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
                                    action_kind=(updated_step.decision.action if updated_step.decision else decision.action),
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
                    final_answer = final_text or (updated_step.result.observation if updated_step.result else "") or "Stopped before completing all steps."
                    logger.info("[ReActAgent.handle] Step=%d requested stop -> finalising", plan.index)
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
        logger.info("[ReActAgent.handle] done transcript_lines=%d executed_steps=%d", len(transcript_lines), len(executed_trace))
        return "\n".join(transcript_lines)