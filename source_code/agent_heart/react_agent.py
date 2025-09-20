from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4
import re
import json
import logging

from agent_core.models import Subquery, ToolCall
from agent_core.enums import QueryStatus, SubqueryStatus, ExecutionMode
from agent_memory.pronouns_resolution.pronouns_resolver import PronounsResolver
from agent_memory.conversation_memory import ConversationMemory
from agent_memory.query_memory import QueryMemory
from agent_memory.context_resolution.context_resolver import ContextResolver
from agent_memory.pronouns_resolution.entity_extractor import EntityExtractor
from task_management.tool_executor import ToolExecutor
from task_management.answer_generator import AnswerGenerator
from task_management.answer_assessment.confidence_assessor import ConfidenceAssessor
from tool_management.tools.agent_tool import AgentTool, ToolResponse
from llm_adapters.llm_base import LLMBase
from task_management.tool_query_refiner import ToolQueryRefiner, RefinerPolicy

from agent_core.agent_actions.action import Action
from agent_core.agent_actions.use_tool_action import UseToolAction

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ReActStep:
    index: int
    thought: str
    action_kind: str
    action_tool: Optional[str]
    action_input: str
    observation: str


@dataclass
class ReActAgentConfig:
    quantitative_regex: str = r"(?:\bpercent(?:age)?\b|\b%(\s|$)|\bshare\b|\bratio\b|\bproportion\b|\bcompare(?:d)?\b|\bversus\b|\bvs\.?\b|\bpopulation\b|\bcalculate\b|\bcompute\b|\btotal\b|\bnumber\b)"
    max_steps: int = 10
    confidence_threshold: float = 0.80
    max_recent_steps_in_prompt: int = 6
    max_evidence_snippets: int = 5
    observation_snippet_limit: int = 800
    fact_schema_max_items: int = 6
    enforce_thought_action_alignment: bool = True
    action_alignment_tokeniser_regex: str = r"[A-Za-z0-9]+"
    input_alignment_min_overlap_tokens: int = 2


class ActionRegistry:
    def __init__(self, actions: List[Action]):
        self._actions = list(actions or [])
        logger.info(f"[ActionRegistry.__init__] Registered actions count={len(self._actions)}")

    @property
    def names(self) -> List[str]:
        return [a.name for a in self._actions]

    def as_catalog_text(self) -> str:
        lines = []
        for a in self._actions:
            lines.append(a.to_catalog_entry())
        text = "\n".join(lines)
        logger.debug(f"[ActionRegistry.as_catalog_text] Catalogue text length={len(text)}")
        return text

    def get(self, name: str) -> Optional[Action]:
        for a in self._actions:
            if a.name == name:
                return a
        return None


class ReActAgent:
    def __init__(
        self,
        planner_llm: LLMBase,
        summariser_llm: LLMBase,
        confidence_llm: LLMBase,
        pronouns_llm: LLMBase,
        entity_extractor: EntityExtractor,
        tools: Dict[str, AgentTool],
        separation_strategy: Optional[Any] = None,
        actions: Optional[List[Action]] = None,
        *,
        max_steps: int = None,
        confidence_threshold: float = None,
        entity_store: Optional[Any] = None,
        config: Optional[ReActAgentConfig] = None,
    ):
        if not tools:
            raise ValueError("ReActAgent requires at least one tool")
        self.planner_llm = planner_llm
        self.summariser_llm = summariser_llm
        self.confidence_llm = confidence_llm
        self.tools = tools
        self.tool_executor = ToolExecutor(tools)
        self.answer_generator = AnswerGenerator(summariser_llm)
        self.confidence_assessor = ConfidenceAssessor(confidence_llm)
        self.config = config or ReActAgentConfig()
        if max_steps is not None:
            self.config.max_steps = int(max_steps)
        if confidence_threshold is not None:
            self.config.confidence_threshold = float(confidence_threshold)
        resolver = PronounsResolver(llm=pronouns_llm, max_ctx_chars=4000, max_candidates=12, entity_store=entity_store)
        setattr(resolver, "entity_extractor", entity_extractor)
        self.pronouns_resolver = resolver
        self.entity_extractor = entity_extractor
        self.conversation_memory = ConversationMemory(pronouns_resolver=resolver)
        self.query_memory = QueryMemory(pronouns_resolver=resolver)
        self.context_resolver = ContextResolver(llm=summariser_llm, entity_extractor=entity_extractor, entity_store=entity_store)
        self.tool_query_refiner = ToolQueryRefiner(llm=self.summariser_llm, max_chars=300)
        self.separation_strategy = separation_strategy
        if self.separation_strategy is None:
            logger.info("[ReActAgent.__init__] Split-query strategy is disabled; planning proceeds stepwise via thought elicitation")
        if actions:
            self.action_registry = ActionRegistry(actions)
        else:
            self.action_registry = ActionRegistry([UseToolAction(available_tools=list(self.tools.keys()), tool_registry=self.tools)])
        self.fact_store: Dict[str, Any] = {}
        logger.info(f"[ReActAgent.__init__] Initialised with tools={list(self.tools.keys())}, actions={self.action_registry.names}, max_steps={self.config.max_steps}, confidence_threshold={self.config.confidence_threshold:.2f}")
        logger.info("[ReActAgent.__init__] ToolQueryRefiner wired with summariser_llm as refinement backbone")

    def handle(self, question: str) -> str:
        logger.info(f"[ReActAgent.handle] Enter question length={len(question)}")
        try:
            self.pronouns_resolver.start_new_query()
            logger.debug("[ReActAgent.handle] Pronouns resolver cycle initialised")
        except Exception as e:
            logger.warning(f"[ReActAgent.handle] Pronouns resolver start failed error={e!r}")
        self.query_memory.start_new(question)
        if self.query_memory.current_query_model:
            self.query_memory.current_query_model.status = QueryStatus.RUNNING
            self.conversation_memory.set_current_query_model(self.query_memory.current_query_model)
            logger.info("[ReActAgent.handle] Query memory set to RUNNING and linked to conversation memory")
        transcript_lines: List[str] = []
        transcript_lines.append(f"You asked: {self._sanitise_to_text(question)}")
        steps: List[ReActStep] = []
        context_snippets: List[str] = []
        working_query = self._sanitise_to_text(question)
        for step_index in range(1, self.config.max_steps + 1):
            guidance = self._elicit_step_thought(working_query, steps, context_snippets)
            next_focus = self._parse_next_focus(guidance)
            thought_text = guidance.get("now_reasoning") or guidance.get("next_focus") or "I will identify the smallest next step."
            transcript_lines.append(self._format_thought_line(step_index, thought_text))

            plan = self._plan_next_step_actions(working_query, steps, context_snippets, next_focus)
            action = (plan.get("action") or "").upper()
            tool_name = plan.get("tool_name", "")
            tool_input = plan.get("tool_input", "")
            logger.info(f"[ReActAgent.handle] Planner returned action={action!r} tool_name={tool_name!r} tool_input_len={len(tool_input)} at step_index={step_index}")
            if action == "STOP":
                transcript_lines.append(self._format_action_line_human(step_index, action, tool_name, tool_input, message="Stopping here as requested."))
                self._finalise_in_memory(question, "Session stopped.")
                return "\n".join(transcript_lines)
            if action == "CLARIFY":
                message = plan.get("message") or "Could you clarify?"
                transcript_lines.append(self._format_action_line_human(step_index, action, tool_name, tool_input, message=message))
                self._finalise_in_memory(question, message)
                return "\n".join(transcript_lines)
            if action == "ANSWER_BY_ITSELF":
                final_text = plan.get("final_answer") or self._synthesise_final_answer(question, steps, context_snippets)
                ok = self._is_confident_enough(question, final_text, None)
                logger.info(f"[ReActAgent.handle] Self-answer confidence ok={ok}")
                if ok:
                    transcript_lines.append(self._format_action_line_human(step_index, "FINALISE", "", "", final_answer=final_text))
                    self._finalise_in_memory(question, final_text)
                    return "\n".join(transcript_lines)
                action = "USE_TOOL"
            if action == "FINALISE":
                if self._is_ready_to_finalise(working_query, steps, guidance, context_snippets):
                    final_text = plan.get("final_answer") or self._synthesise_final_answer(question, steps, context_snippets)
                    transcript_lines.append(self._format_action_line_human(step_index, "FINALISE", "", "", final_answer=final_text))
                    self._finalise_in_memory(question, final_text)
                    return "\n".join(transcript_lines)
                logger.info("[ReActAgent.handle] Guard prevented premature FINALISE; switching to USE_TOOL via guidance")
                action = "USE_TOOL"
                tool_choice = self._derive_tool_decision_via_llm(working_query, guidance)
                tool_name = tool_choice.get("tool_name") or tool_name
                tool_input = tool_choice.get("tool_input") or tool_input
            if action == "USE_TOOL":
                if not tool_name or tool_name not in self.tools:
                    logger.info(f"[ReActAgent.handle] Tool filled by ranking selection tool={tool_name!r}")
                transcript_lines.append(self._format_action_line_human(step_index, action, tool_name, tool_input))
                policy: Optional[RefinerPolicy] = None
                obs_text, evidence, raw_text = self._execute_and_observe_with_policy(tool_name, tool_input, question, policy, guidance)
                transcript_lines.append(self._format_observation_line(step_index, obs_text))
                steps.append(ReActStep(index=step_index, thought=thought_text, action_kind="USE_TOOL", action_tool=tool_name, action_input=tool_input, observation=obs_text))
                if evidence:
                    context_snippets.append(evidence)
                if raw_text:
                    self._update_fact_store_from_result(raw_text, question)
                confidence_value_after_tool = 0.0
                try:
                    confidence_value_after_tool = self.confidence_assessor.assess(query=question, result=raw_text, answer=obs_text)
                    logger.info(f"[ReActAgent.handle] Confidence after tool execution={confidence_value_after_tool:.3f}")
                except Exception as e:
                    logger.warning(f"[ReActAgent.handle] Confidence assessor raised error={e!r}; treating as low confidence")
                    confidence_value_after_tool = 0.0
                if confidence_value_after_tool < self.config.confidence_threshold:
                    logger.info(f"[ReActAgent.handle] Confidence {confidence_value_after_tool:.3f} below threshold {self.config.confidence_threshold:.3f}; entering recovery loop")
                    max_recovery_tries = 4
                    recovery_attempt = 0
                    latest_obs_text = obs_text
                    latest_raw_text = raw_text
                    while True:
                        if confidence_value_after_tool >= self.config.confidence_threshold:
                            logger.info(f"[ReActAgent.handle] Confidence now {confidence_value_after_tool:.3f} (>= threshold); exiting recovery loop")
                            break
                        if recovery_attempt >= max_recovery_tries:
                            logger.info(f"[ReActAgent.handle] Reached max_recovery_tries={max_recovery_tries}; exiting recovery loop")
                            break
                        tried_tool_names = []
                        for s in steps:
                            if s.action_kind.upper() == "USE_TOOL" and s.action_tool:
                                tried_tool_names.append(str(s.action_tool))
                        tried_tool_names = list(dict.fromkeys(tried_tool_names))
                        logger.info(f"[ReActAgent.handle] Recovery attempt {recovery_attempt+1}; tried_tools={tried_tool_names}")
                        recovery_decision = self._select_recovery_tool_via_llm(next_focus, tried_tool_names)
                        chosen_recovery_tool = self._sanitise_to_text(recovery_decision.get("tool_name", ""))
                        chosen_recovery_input = self._sanitise_to_text(recovery_decision.get("tool_input", "")) or tool_input or working_query
                        chosen_recovery_decision = (recovery_decision.get("decision") or "").strip().lower()
                        if chosen_recovery_decision == "use" and chosen_recovery_tool and chosen_recovery_tool in self.tools and chosen_recovery_tool not in tried_tool_names:
                            logger.info(f"[ReActAgent.handle] Recovery selector chose available tool={chosen_recovery_tool!r}; executing with input length={len(chosen_recovery_input)}")
                            transcript_lines.append(self._format_action_line_human(step_index, action, chosen_recovery_tool, chosen_recovery_input))
                            obs2, ev2, raw2 = self._execute_and_observe_with_policy(chosen_recovery_tool, chosen_recovery_input, question, None, guidance)
                            transcript_lines.append(self._format_observation_line(step_index, obs2))
                            steps.append(ReActStep(index=step_index, thought=thought_text, action_kind="USE_TOOL", action_tool=chosen_recovery_tool, action_input=chosen_recovery_input, observation=obs2))
                            if ev2:
                                context_snippets.append(ev2)
                            if raw2:
                                self._update_fact_store_from_result(raw2, question)
                            latest_obs_text, evidence, latest_raw_text = obs2, ev2, raw2
                            try:
                                confidence_value_after_tool = self.confidence_assessor.assess(query=question, result=latest_raw_text, answer=latest_obs_text)
                                logger.info(f"[ReActAgent.handle] Confidence after recovery attempt {recovery_attempt+1} = {confidence_value_after_tool:.3f}")
                            except Exception as e:
                                logger.warning(f"[ReActAgent.handle] Confidence assessor error after recovery attempt {recovery_attempt+1}: {e!r}; keeping previous confidence")
                            recovery_attempt += 1
                            continue
                        if chosen_recovery_decision != "use":
                            logger.info("[ReActAgent.handle] Recovery selector advised not to use remaining tools; exiting loop")
                        elif chosen_recovery_tool and chosen_recovery_tool not in self.tools:
                            logger.warning(f"[ReActAgent.handle] Recovery suggested unavailable tool={chosen_recovery_tool!r}; exiting loop")
                        elif chosen_recovery_tool in tried_tool_names:
                            logger.info(f"[ReActAgent.handle] Recovery suggested already tried tool={chosen_recovery_tool!r}; exiting loop")
                        break
                    obs_text = latest_obs_text
                    raw_text = latest_raw_text
                if self._is_quantitative_query(working_query) and not self._is_ready_to_finalise(working_query, steps, guidance, context_snippets):
                    logger.info("[ReActAgent.handle] Quantitative guard indicates more evidence is needed; continuing planning loop")
                    continue
                if self._is_confident_enough(question, obs_text, raw_text):
                    final_text = self._synthesise_final_answer(question, steps, context_snippets)
                    transcript_lines.append(self._format_thought_line(step_index + 1, "I have enough information to answer now."))
                    transcript_lines.append(self._format_action_line_human(step_index + 1, "FINALISE", "", "", final_answer=final_text))
                    self._finalise_in_memory(question, final_text)
                    return "\n".join(transcript_lines)
                continue
            transcript_lines.append(self._format_action_line_human(step_index, "SEARCH", "", working_query))
            obs, ev, raw_text = self._execute_and_observe(self._heuristic_tool_selection(guidance.get("next_focus", ""), working_query), working_query, working_query)
            transcript_lines.append(self._format_observation_line(step_index, obs))
            steps.append(ReActStep(index=step_index, thought=thought_text or "Search fallback.", action_kind="SEARCH", action_tool="(heuristic)", action_input=working_query, observation=obs))
            if ev:
                context_snippets.append(ev)
            if raw_text:
                self._update_fact_store_from_result(raw_text, question)
            if self._is_confident_enough(question, obs, raw_text):
                final_text = self._synthesise_final_answer(question, steps, context_snippets)
                transcript_lines.append(self._format_thought_line(step_index + 1, "The evidence looks sufficient to answer."))
                transcript_lines.append(self._format_action_line_human(step_index + 1, "FINALISE", "", "", final_answer=final_text))
                self._finalise_in_memory(question, final_text)
                return "\n".join(transcript_lines)
        final_best = self._synthesise_final_answer(question, steps, context_snippets)
        transcript_lines.append(self._format_thought_line(self.config.max_steps + 1, "I will give the best possible answer with what I have."))
        transcript_lines.append(self._format_action_line_human(self.config.max_steps + 1, "FINALISE", "", "", final_answer=final_best))
        self._finalise_in_memory(question, final_best)
        return "\n".join(transcript_lines)

    def _elicit_step_thought(self, question: str, steps: List[ReActStep], context_snippets: List[str]) -> Dict[str, Any]:
        recent = []
        for s in steps[-5:]:
            recent.append(f"Thought: {s.thought}\nAction: {s.action_kind} [{s.action_tool or ''} :: {s.action_input}]\nObservation: {s.observation}")
        evidence_block = "\n".join(context_snippets[-4:]) if context_snippets else "(none)"
        schema = '{"needed_facts":["<list>"],"resolved_facts":["<list>"],"pending_facts":["<list>"],"next_focus":"<one short imperative>","now_reasoning":"<one sentence in first person>"}'
        prompt = (
            "You are planning the next precise step.\n"
            "Knowledge cut-off is 2023. The user expects current, reliable, and precise information. Prefer tools for numbers and verifiable facts.\n"
            "Answer strictly in JSON matching the schema below.\n\n"
            f"Question: {question}\n\n"
            f"Recent trace:\n{('\n---\n'.join(recent)) if recent else '(none)'}\n\n"
            f"Observational evidence snippets:\n{evidence_block}\n\n"
            "Task:\n"
            "1) Make query easier to handle- resolve pronouns inside of query or resolve the context .\n"
            "2) List the minimal facts required to answer the question as noun phrases.\n"
            "3) Mark which facts are already resolved given the observations; the rest are pending.\n"
            "4) Decide the single most useful next focus as an imperative to retrieve information about the first not marked fact(e.g., 'retrieve <whole fact>' or 'compute <whole fact>'). The context and relation are very important. \n"
            "5) If all facts are resolved, in the next focus join the results of observations to resolve contextually missing facts of the question.\n"
            "6) Provide one sentence in first person explaining why this is the next step.\n\n"
            f"Schema: {schema}\n"
            "JSON:"
        )
        logger.info(f"[ReActAgent._elicit_step_thought] PROMPT: {prompt}")
        raw = (self.planner_llm.generate(prompt) or "").strip()
        logger.info(f"[ReActAgent._elicit_step_thought] RAW {raw}")
        data = self._safe_parse_json_object(raw)
        data["now_reasoning"] = self._sanitise_to_text(data.get("now_reasoning", "I will proceed with the smallest missing prerequisite."))
        data["next_focus"] = self._sanitise_to_text(data.get("next_focus", "retrieve the key missing fact"))
        if not isinstance(data.get("needed_facts"), list):
            data["needed_facts"] = []
        if not isinstance(data.get("resolved_facts"), list):
            data["resolved_facts"] = []
        if not isinstance(data.get("pending_facts"), list):
            data["pending_facts"] = []
        logger.info(f"[ReActAgent._elicit_step_thought] Parsed next_focus={data['next_focus']!r} needed_count={len(data['needed_facts'])} pending_count={len(data['pending_facts'])}")
        return data

    def _parse_next_focus(self, guidance: Any) -> str:
        if isinstance(guidance, dict):
            value = self._sanitise_to_text(guidance.get("next_focus", ""))
            logger.info(f"The next_focus parsed: {value}")
        else:
            value = self._sanitise_to_text(guidance)
        return value or "retrieve the key missing fact"

    def _plan_next_step_actions(self, question: str, steps: List[ReActStep], context_snippets: List[str], next_focus: Any) -> Dict[str, Any]:
        history_lines = []
        for s in steps[-self.config.max_recent_steps_in_prompt:]:
            history_lines.append(f"Thought: {s.thought}\nAction: {s.action_kind} [{s.action_tool or ''} :: {s.action_input}]\nObservation: {s.observation}")
        context_block = "\n".join(context_snippets[-self.config.max_evidence_snippets:]) if context_snippets else "(none)"
        action_catalog = self.action_registry.as_catalog_text()
        tool_catalog = self._tool_catalog_text()
        allowed_actions_line = f"Allowed ACTION values: {', '.join([n for n in self.action_registry.names if n != 'SPLIT_QUERY'])}"
        schema = '{"think":"<short>","action":"<one of ACTION names>","tool_name":"<required when action=USE_TOOL>","tool_input":"<required when action=USE_TOOL>","message":"<for CLARIFY>","final_answer":"<for FINALISE or ANSWER_BY_ITSELF>"}'
        allowed_tool_names = ", ".join(sorted(self.tools.keys()))
        prompt = (
            f"Plan the next action for the task: {next_focus}. "
            "Return JSON ONLY; no prose outside JSON.\n"
            "Knowledge cut-off is 2023. Prefer tools for numbers and verifiable facts. Do not invent figures. Do not guess.\n"
            "Pick exactly one action from the ACTION CATALOGUE to complete the task:\n"
            f"ACTION CATALOGUE:\n{action_catalog}\n\n"
            "Hard requirements when choosing USE_TOOL:\n"
            "If action='USE_TOOL', you MUST also set 'tool_name' to exactly one name from the TOOL CATALOGUE and you MUST provide a non-empty 'tool_input' that strictly follows that tool's input format and regex. No placeholders, no explanations, no extra text.\n"
            f"Write 'tool_input': {next_focus} \n"
            "If action≠'USE_TOOL', leave 'tool_name' and 'tool_input' empty.\n\n"
            f"TOOL CATALOGUE (valid tool_name values: {allowed_tool_names}):\n{tool_catalog}\n\n"
            f"Schema (use these keys exactly): {schema}\n\n"
            "JSON:"
        )
        raw = (self.planner_llm.generate(prompt) or "").strip()
        logger.info(f"[ReActAgent._plan_next_step_actions] Planner raw:{raw}")
        data = self._safe_parse_json_object(raw)
        data["action"] = (data.get("action") or "").strip().upper()
        data["think"] = self._sanitise_to_text(data.get("think", ""))
        data["tool_name"] = self._sanitise_to_text(data.get("tool_name", ""))
        data["tool_input"] = self._sanitise_to_text(data.get("tool_input", ""))
        data["message"] = self._sanitise_to_text(data.get("message", ""))
        data["final_answer"] = self._sanitise_to_text(data.get("final_answer", ""))
        data = self._normalise_planner_decision(data)
        if data["action"] == "USE_TOOL":
            if not data["tool_input"]:
                logger.info(f"[ReActAgent._plan_next_step_actions] Missing tool_input from planner")
        logger.info(f"[ReActAgent._plan_next_step_actions] Parsed action={data['action']} tool={data.get('tool_name') or None} tool_input_len={len(data.get('tool_input',''))}")
        return data

    def _derive_tool_decision_via_llm(self, question: str, guidance: Dict[str, Any]) -> Dict[str, str]:
        tools_list = [{"name": k, "desc": (v.description or ""), "caps": sorted(list(getattr(v, "capabilities", set()) or []))} for k, v in self.tools.items()]
        schema = '{"tool_name":"<one tool>","tool_input":"<string input for that tool>"}'
        prompt = (
            "Choose the single best tool and its minimal deterministic input for the next focus. JSON only.\n"
            "Prefer tools with computation capability for compute steps; prefer search/lookup/encyclopaedic tools for retrieval.\n"
            "Do not invent numbers. Keep inputs concise and directly executable.\n\n"
            f"Question: {question}\n"
            f"Next focus: {guidance.get('next_focus','')}\n"
            f"Available tools: {json.dumps(tools_list, ensure_ascii=False)}\n"
            f"Schema: {schema}\n"
            "JSON:"
        )
        logger.info(f"[ReActAgent._derive_tool_decision_via_llm] Prompt length={len(prompt)}")
        raw = (self.summariser_llm.generate(prompt) or "").strip()
        logger.info(f"[ReActAgent._derive_tool_decision_via_llm] Raw length={len(raw)}")
        data = self._safe_parse_json_object(raw)
        tool_name = self._sanitise_to_text(data.get("tool_name", ""))
        tool_input = self._sanitise_to_text(data.get("tool_input", ""))
        if tool_name and tool_name not in self.tools:
            ranked = self._rank_tools_for_focus(guidance, question)
            tool_name = ranked[0] if ranked else self._heuristic_tool_selection(guidance.get("next_focus", ""), tool_input or question)
            logger.warning(f"[ReActAgent._derive_tool_decision_via_llm] LLM suggested unavailable tool; selected {tool_name!r} instead")
        if not tool_name:
            ranked = self._rank_tools_for_focus(guidance, question)
            tool_name = ranked[0] if ranked else self._heuristic_tool_selection(guidance.get("next_focus", ""), tool_input or question)
            logger.info(f"[ReActAgent._derive_tool_decision_via_llm] Heuristic selected tool={tool_name!r}")
        if not tool_input:
            tool_input = self._shape_tool_input_for_focus(guidance.get("next_focus", ""), guidance)
            logger.info(f"[ReActAgent._derive_tool_decision_via_llm] Derived minimal input for focus length={len(tool_input)}")
        return {"tool_name": tool_name, "tool_input": tool_input}

    def _rank_tools_for_focus(self, guidance: Dict[str, Any], question: str) -> List[str]:
        focus = (guidance.get("next_focus") or "").lower()
        caps_by_tool = {name: set(getattr(tool, "capabilities", set()) or []) for name, tool in self.tools.items()}
        scores: Dict[str, int] = {}
        for name, caps in caps_by_tool.items():
            score = 0
            if "compute" in focus and any(c in caps for c in ("compute", "math", "arithmetic")):
                score += 5
            if "retrieve" in focus and any(c in caps for c in ("search", "lookup", "encyclopaedia", "crawl", "browse")):
                score += 4
            if any(tok in focus for tok in ("latest", "current", "today", "recent")) and any(c in caps for c in ("search", "news", "browse")):
                score += 3
            if "structured" in caps:
                score += 1
            scores[name] = score
        ranked = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
        logger.info(f"[ReActAgent._rank_tools_for_focus] Ranked tools by focus={focus!r} -> {[(n, scores[n]) for n in ranked]}")
        return ranked

    def _execute_and_observe_with_policy(self, tool_name: str, tool_input: str, user_question: str, policy: Optional[RefinerPolicy], guidance: Optional[Dict[str, Any]] = None) -> Tuple[str, str, str]:
        logger.info(f"[ReActAgent._execute_and_observe_with_policy] Executing tool={tool_name!r} candidate_input_length={len(tool_input)}")
        if tool_name not in self.tools:
            logger.warning(f"[ReActAgent._execute_and_observe_with_policy] Unknown tool={tool_name!r}, applying heuristic")
            tool_name = self._heuristic_tool_selection(guidance.get("next_focus", "") if guidance else "", tool_input or user_question)
        selected_tool = self.tools[tool_name]
        raw_candidate = self._sanitise_to_text(tool_input) or self._sanitise_to_text(user_question)
        try:
            refined_candidate = self.tool_query_refiner.refine(raw_candidate, selected_tool, policy=policy)
            logger.info(f"[ReActAgent._execute_and_observe_with_policy] Refiner produced tool_input length={len(refined_candidate)} for tool={tool_name}")
        except Exception as e:
            logger.warning(f"[ReActAgent._execute_and_observe_with_policy] Refiner failed error={e!r}, using fallback")
            refined_candidate = raw_candidate
        refined_input = self._sanitise_to_text(refined_candidate)
        try:
            result = self.tool_executor.execute(tool_name, refined_input)
            logger.info(f"[ReActAgent._execute_and_observe_with_policy] Tool executed successfully name={tool_name}")
        except Exception as e:
            logger.warning(f"[ReActAgent._execute_and_observe_with_policy] Tool executor raised error={e!r}")
            observation = f"The tool {tool_name} failed; I will try a different approach."
            return observation, "", ""
        tool_call = ToolCall(name=tool_name, input=refined_input)
        try:
            tool_call.response = {"raw": getattr(result, "raw", None), "text": str(result)}
        except Exception:
            tool_call.response = {"text": str(result)}
        if self.query_memory.current_query_model:
            sub = Subquery(id=str(uuid4()), text=user_question, mode=ExecutionMode.NEEDS_TOOLS, status=SubqueryStatus.ANSWERED)
            sub.answer = str(result)
            sub.tool_calls.append(tool_call)
            self.query_memory.current_query_model.subqueries.append(sub)
            logger.debug("[ReActAgent._execute_and_observe_with_policy] Subquery model updated with tool call and answer")
        short_obs = self._summarise_observation(user_question, result)
        evidence_snippet = self._extract_evidence_snippet(result)
        try:
            raw_text = getattr(result, "text", None) or str(result)
        except Exception:
            raw_text = short_obs
        logger.info(f"[ReActAgent._execute_and_observe_with_policy] Observation length={len(short_obs)} evidence_len={len(evidence_snippet)} raw_len={len(raw_text)}")
        return short_obs, evidence_snippet, raw_text

    def _execute_and_observe(self, tool_name: str, tool_input: str, user_question: str) -> Tuple[str, str, str]:
        logger.info(f"[ReActAgent._execute_and_observe] Executing tool={tool_name!r} input_length={len(tool_input)}")
        if tool_name not in self.tools:
            logger.warning(f"[ReActAgent._execute_and_observe] Unknown tool={tool_name!r}, substituting heuristic tool")
            tool_name = self._heuristic_tool_selection("", tool_input or user_question)
        selected_tool = self.tools[tool_name]
        raw_candidate = self._sanitise_to_text(tool_input) or self._sanitise_to_text(user_question)
        try:
            refined_candidate = self.tool_query_refiner.refine(raw_candidate, selected_tool, policy=None)
            logger.info(f"[ReActAgent._execute_and_observe] Refiner produced tool_input length={len(refined_candidate)} for tool={tool_name}")
        except Exception as e:
            logger.warning(f"[ReActAgent._execute_and_observe] Refiner failed error={e!r}, falling back to sanitised candidate")
            refined_candidate = raw_candidate
        refined_input = self._sanitise_to_text(refined_candidate)
        try:
            result = self.tool_executor.execute(tool_name, refined_input)
            logger.info(f"[ReActAgent._execute-and_observe] Tool executed successfully name={tool_name}")
        except Exception as e:
            logger.warning(f"[ReActAgent._execute_and_observe] Tool executor raised error={e!r}")
            observation = f"The tool {tool_name} failed; I will try a different approach."
            return observation, "", observation
        tool_call = ToolCall(name=tool_name, input=refined_input)
        try:
            tool_call.response = {"raw": getattr(result, "raw", None), "text": str(result)}
        except Exception:
            tool_call.response = {"text": str(result)}
        if self.query_memory.current_query_model:
            sub = Subquery(id=str(uuid4()), text=user_question, mode=ExecutionMode.NEEDS_TOOLS, status=SubqueryStatus.ANSWERED)
            sub.answer = str(result)
            sub.tool_calls.append(tool_call)
            self.query_memory.current_query_model.subqueries.append(sub)
            logger.debug("[ReActAgent._execute_and_observe] Subquery model updated with tool call and answer")
        short_obs = self._summarise_observation(user_question, result)
        evidence_snippet = self._extract_evidence_snippet(result)
        try:
            raw_text = getattr(result, "text", None) or str(result)
        except Exception:
            raw_text = short_obs
        logger.info(f"[ReActAgent._execute_and_observe] Observation length={len(short_obs)} evidence_len={len(evidence_snippet)} raw_len={len(raw_text)}")
        return short_obs, evidence_snippet, raw_text

    def _summarise_observation(self, user_question: str, result: ToolResponse) -> str:
        try:
            raw_text = getattr(result, "text", None) or str(result)
        except Exception:
            raw_text = str(result)
        prompt = (
            "Summarise the tool result into one or two concise sentences that directly relate to the question. "
            "If nothing relevant was found, say so and propose the most likely next retrieval focus. British English.\n\n"
            f"Question: {user_question}\n\n"
            f"Tool Result:\n{raw_text}\n\n"
            "Summary:"
        )
        out = (self.summariser_llm.generate(prompt) or "").strip()
        value = self._sanitise_to_text(out) or "No immediately relevant information found."
        logger.debug(f"[ReActAgent._summarise_observation] Summary length={len(value)}")
        return value

    def _extract_evidence_snippet(self, result: ToolResponse) -> str:
        try:
            raw = getattr(result, "raw", None)
            if isinstance(raw, str) and raw.strip():
                snippet = raw.strip()
                if len(snippet) > self.config.observation_snippet_limit:
                    snippet_trunc = snippet[: self.config.observation_snippet_limit]
                    logger.debug("[ReActAgent._extract_evidence_snippet] Raw snippet truncated")
                    return snippet_trunc
                return snippet
        except Exception as e:
            logger.debug(f"[ReActAgent._extract_evidence_snippet] Access raw failed error={e!r}")
        try:
            text = getattr(result, "text", None) or str(result)
            if text and len(text) > self.config.observation_snippet_limit:
                logger.debug("[ReActAgent._extract_evidence_snippet] Text snippet truncated")
                return text[: self.config.observation_snippet_limit]
            return text or ""
        except Exception as e:
            logger.debug(f"[ReActAgent._extract_evidence_snippet] Fallback to str(result) failed error={e!r}")
        return ""

    def _update_fact_store_from_result(self, raw_text: str, question: str) -> None:
        schema = '{"facts":[{"key":"<short>","value":"<string>","number":"<optional number or null>","unit":"<optional>"}]}'
        prompt = (
            "Extract up to N compact facts from the text that are directly useful to answer the question. "
            "Return JSON only. If a numeric value is present, include it as number and set unit if obvious. "
            "Keep keys short and specific to the entity or attribute.\n\n"
            f"N={self.config.fact_schema_max_items}\n"
            f"Question: {question}\n\n"
            f"Text:\n{raw_text}\n\n"
            f"Schema: {schema}\n"
            "JSON:"
        )
        raw = (self.summariser_llm.generate(prompt) or "").strip()
        data = self._safe_parse_json_object(raw)
        facts = data.get("facts", [])
        added = 0
        for item in facts if isinstance(facts, list) else []:
            key = self._sanitise_to_text(item.get("key", ""))
            val = self._sanitise_to_text(item.get("value", ""))
            num = item.get("number", None)
            unit = self._sanitise_to_text(item.get("unit", ""))
            if key and (val or num is not None):
                self.fact_store[key] = {"value": val, "number": num, "unit": unit}
                added += 1
        logger.info(f"[ReActAgent._update_fact_store_from_result] Facts added={added} total={len(self.fact_store)}")

    def _is_quantitative_query(self, question: str) -> bool:
        q = (question or "").lower()
        match = bool(re.search(self.config.quantitative_regex, q))
        logger.debug(f"[ReActAgent._is_quantitative_query] quantitative={match}")
        return match

    def _is_ready_to_finalise(self, question: str, steps: List[ReActStep], guidance: Dict[str, Any], context_snippets: List[str]) -> bool:
        quantitative = self._is_quantitative_query(question)
        if not quantitative:
            logger.debug("[ReActAgent._is_ready_to_finalise] Non-quantitative query; allowing finalise")
            return True
        pending = guidance.get("pending_facts") or []
        if pending:
            logger.info(f"[ReActAgent._is_ready_to_finalise] Pending facts remain count={len(pending)}; not ready")
            return False
        return True

    def _is_confident_enough(self, question: str, candidate_answer: str, tool_result_text: Optional[str]) -> bool:
        try:
            score = self.confidence_assessor.assess(query=question, result=tool_result_text, answer=candidate_answer)
        except Exception as e:
            logger.warning(f"[ReActAgent._is_confident_enough] ConfidenceAssessor failed error={e!r}")
            score = 0.0
        logger.info(f"[ReActAgent._is_confident_enough] Confidence score={score:.3f} threshold={self.config.confidence_threshold:.3f}")
        return score >= self.config.confidence_threshold

    def _heuristic_tool_selection(self, focus_or_action: str, tool_input: str) -> str:
        focus = (focus_or_action or "").lower()
        for name, t in self.tools.items():
            caps = set(getattr(t, "capabilities", set()) or [])
            if "compute" in focus and any(c in caps for c in ("compute", "math", "arithmetic")):
                logger.info(f"[ReActAgent._heuristic_tool_selection] Heuristic chose {name!r} for compute focus")
                return name
        for name, t in self.tools.items():
            caps = set(getattr(t, "capabilities", set()) or [])
            if any(c in caps for c in ("search", "lookup", "encyclopaedia", "crawl", "browse", "news")):
                logger.info(f"[ReActAgent._heuristic_tool_selection] Heuristic chose {name!r} for retrieval")
                return name
        any_tool = next(iter(self.tools.keys()))
        logger.info(f"[ReActAgent._heuristic_tool_selection] Fallback to {any_tool!r}")
        return any_tool

    def _synthesise_final_answer(self, original_question: str, steps: List[ReActStep], context_snippets: List[str]) -> str:
        parts = []
        for s in steps:
            parts.append(f"- Thought {s.index}: {s.thought}\n  Action {s.index}: {s.action_kind} [{s.action_tool or ''} :: {s.action_input}]\n  Observation {s.index}: {s.observation}")
        ctx = "\n".join(context_snippets[-5:]) if context_snippets else "(none)"
        prompt = (
            "Combine the observations into a direct, final answer to the user's question. "
            "Cite the key fact in plain language without formal references. British English, one or two sentences.\n\n"
            f"Question:\n{original_question}\n\n"
            f"Trace:\n{'\n'.join(parts) if parts else '(none)'}\n\n"
            f"Context:\n{ctx}\n\n"
            "Final Answer:"
        )
        out = (self.summariser_llm.generate(prompt) or "").strip()
        value = self._sanitise_to_text(out) or "I could not determine the answer conclusively."
        logger.info(f"[ReActAgent._synthesise_final_answer] Final answer length={len(value)}")
        return value

    def _finalise_in_memory(self, question: str, final_answer: str) -> None:
        if self.query_memory.current_query_model:
            self.query_memory.current_query_model.final_answer = final_answer
            self.query_memory.current_query_model.status = QueryStatus.ANSWERED
        self.conversation_memory.update(question, final_answer)
        logger.info(f"[ReActAgent._finalise_in_memory] Final answer stored length={len(final_answer)}")

    def _format_thought_line(self, index: int, text: str) -> str:
        t = self._sanitise_to_text(text) or "I am considering the best next step."
        if not re.match(r"^(i|i’m|i am|let me|i should|i will|i think|now i)\b", t.lower()):
            t = f"I’m thinking: {t}"
        t = self._ensure_sentence(t)
        line = f"Thought {index}: {t}"
        logger.debug(f"[ReActAgent._format_thought_line] {line}")
        return line

    def _format_action_line_human(self, index: int, action: str, tool_name: str, tool_input: str, message: Optional[str] = None, subqueries: Optional[List[str]] = None, final_answer: Optional[str] = None) -> str:
        a = action.upper().strip()
        tn = (tool_name or "").strip()
        ti = self._sanitise_to_text(tool_input)
        if a == "USE_TOOL":
            if tn:
                text = f"I’ll use {tn} with: {ti or 'default input'}."
            else:
                text = f"I’ll use the most suitable tool with: {ti or 'default input'}."
            line = f"Act {index}: {self._ensure_sentence(text)}"
            logger.debug(f"[ReActAgent._format_action_line_human] {line}")
            return line
        if a == "CLARIFY":
            msg = self._ensure_sentence(message or "Could you clarify?")
            line = f"Act {index}: I’ll ask you to clarify: {msg}"
            logger.debug(f"[ReActAgent._format_action_line_human] {line}")
            return line
        if a == "FINALISE":
            ans = self._sanitise_to_text(final_answer or "")
            line = f"Act {index}: Final answer: [{ans}]"
            logger.debug(f"[ReActAgent._format_action_line_human] {line}")
            return line
        if a == "ANSWER_BY_ITSELF":
            line = f"Act {index}: I can answer this directly."
            logger.debug(f"[ReActAgent._format_action_line_human] {line}")
            return line
        if a == "STOP":
            line = f"Act {index}: {self._ensure_sentence(message or 'Stopping here.')}"
            logger.debug(f"[ReActAgent._format_action_line_human] {line}")
            return line
        if a == "SEARCH":
            text = f"I’ll search for: {ti or 'the most relevant information'}."
            line = f"Act {index}: {self._ensure_sentence(text)}"
            logger.debug(f"[ReActAgent._format_action_line_human] {line}")
            return line
        line = f"Act {index}: {self._ensure_sentence(a.title())}"
        logger.debug(f"[ReActAgent._format_action_line_human] {line}")
        return line

    def _format_observation_line(self, index: int, text: str) -> str:
        t = self._sanitise_to_text(text) or "No clear result."
        if not re.match(r"^(i|i’ve|i have|here|this)\b", t.lower()):
            t = f"I found: {t}"
        t = self._ensure_sentence(t)
        line = f"Observation {index}: {t}"
        logger.debug(f"[ReActAgent._format_observation_line] {line}")
        return line

    def _ensure_sentence(self, s: str) -> str:
        x = self._sanitise_to_text(s)
        if not x:
            return ""
        if x[-1] not in ".!?":
            x = x + "."
        return x

    def _normalise_planner_decision(self, data: Dict[str, Any]) -> Dict[str, Any]:
        valid_actions = {n.upper() for n in self.action_registry.names}
        action_upper = (data.get("action") or "").strip().upper()
        tool_names_lower = {k.lower() for k in self.tools.keys()}
        if action_upper == "SPLIT_QUERY":
            data["action"] = "USE_TOOL"
            logger.info("[ReActAgent._normalise_planner_decision] SPLIT_QUERY is disabled; coercing to USE_TOOL")
            return data
        if action_upper not in valid_actions:
            alias = action_upper.replace("-", "_").replace(" ", "_")
            hinted_tool = (data.get("tool_name") or "").strip().lower()
            if hinted_tool in tool_names_lower:
                data["action"] = "USE_TOOL"
                data["tool_name"] = hinted_tool
                logger.info(f"[ReActAgent._normalise_planner_decision] Unknown action '{action_upper}' but valid tool hint detected; forcing USE_TOOL with tool_name='{hinted_tool}'")
                return data
            data["action"] = "USE_TOOL"
            logger.warning(f"[ReActAgent._normalise_planner_decision] Unknown action '{action_upper}', defaulting to USE_TOOL without explicit tool_name")
            return data
        if data["action"] == "USE_TOOL":
            hinted_tool = (data.get("tool_name") or "").strip().lower()
            if hinted_tool and hinted_tool not in tool_names_lower:
                logger.warning(f"[ReActAgent._normalise_planner_decision] Provided tool_name '{hinted_tool}' not available; will map later")
        return data

    def _sanitise_to_text(self, s: Any) -> str:
        try:
            if isinstance(s, (dict, list, tuple)):
                s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
        s2 = (str(s) if s is not None else "").replace("\n", " ").replace("\r", " ").strip()
        value = re.sub(r"\s{2,}", " ", s2)
        logger.debug(f"[ReActAgent._sanitise_to_text] Sanitised text length={len(value)}")
        return value

    def _safe_parse_json_object(self, raw: str) -> Dict[str, Any]:
        s = (raw or "").strip()
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)
        if "{" in s and "}" in s:
            s = s[s.find("{"): s.rfind("}") + 1]
        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s, flags=re.IGNORECASE)
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                logger.debug("[ReActAgent._safe_parse_json_object] Parsed JSON object successfully")
                return obj
        except Exception as e:
            logger.warning(f"[ReActAgent._safe_parse_json_object] json.loads failed error={e!r}")
        logger.warning("[ReActAgent._safe_parse_json_object] Could not parse JSON, returning default Use_Tool plan")
        return {"think": "Begin with a targeted retrieval.", "action": "USE_TOOL", "tool_name": "", "tool_input": ""}

    def _select_recovery_tool_via_llm(self, next_focus: Any, tried_tools: List[str]) -> Dict[str, str]:
        logger.info(f"Tool recovery next_focus: {next_focus}")
        tried_lower = {t.lower() for t in tried_tools if t}
        remaining_tools = {name: tool for name, tool in self.tools.items() if name.lower() not in tried_lower}
        remaining_list = [{"name": n, "desc": (t.description or ""), "caps": sorted(list(getattr(t, "capabilities", set()) or []))} for n, t in remaining_tools.items()]
        tried_list = [{"name": n} for n in tried_tools]
        schema = '{"decision":"<use|none>","tool_name":"<name or empty>","tool_input":"<string>","reason":"<short>"}'
        tool_catalog = self._tool_catalog_text()
        prompt = (
            f"Confidence after the last tool run is below threshold. You must decide whether any of the REMAINING tools should be tried to complete the task: {next_focus}.\n"
            "Return JSON only.\n"
            "If one of the remaining tools is suitable, set decision='use', choose that tool_name from the remaining list, and provide the minimal deterministic tool_input.\n"
            "If none of the remaining tools are likely to help, set decision='none' and provide a hypothetical tool_name that would help in principle.\n"
            "Do not invent numbers. Keep inputs concise and directly executable.\n\n"
            f"Tried tools: {json.dumps(tried_list, ensure_ascii=False)}\n"
            f"Remaining tools: {json.dumps(remaining_list, ensure_ascii=False)}\n"
            f"Choose tool based on tool description: {tool_catalog}\n"
            f"Schema: {schema}\n"
            f"Write 'tool_input' in schema: {next_focus} \n"
            "JSON:"
        )
        logger.info(f"[ReActAgent._select_recovery_tool_via_llm] Prompt length={len(prompt)} remaining_count={len(remaining_list)} tried_count={len(tried_list)}")
        raw = (self.summariser_llm.generate(prompt) or "").strip()
        logger.info(f"Tool recovery result: {raw}")
        logger.info(f"[ReActAgent._select_recovery_tool_via_llm] Raw length={len(raw)}")
        data = self._safe_parse_json_object(raw)
        decision = (data.get("decision") or "").strip().lower()
        tool_name = self._sanitise_to_text(data.get("tool_name", ""))
        tool_input = self._sanitise_to_text(data.get("tool_input", ""))
        reason = self._sanitise_to_text(data.get("reason", ""))
        logger.info(f"[ReActAgent._select_recovery_tool_via_llm] Parsed decision={decision!r} tool_name={tool_name!r} input_len={len(tool_input)} reason={reason!r}")
        return {"decision": decision, "tool_name": tool_name, "tool_input": tool_input, "reason": reason}

    def _tool_catalog_text(self) -> str:
        lines: List[str] = []
        for name, tool in self.tools.items():
            try:
                desc = (getattr(tool, "description", "") or "").strip().replace("\n", " ")
                instr = getattr(tool, "refiner_instructions", "") or ""
                fmt = getattr(tool, "refiner_input_format", "") or ""
                rx = getattr(tool, "refiner_input_regex", None) or ""
                forb = getattr(tool, "refiner_forbidden", "") or ""
                caps = " ".join(sorted(list(getattr(tool, "capabilities", set()) or [])))
                examples_block = (getattr(tool, "refiner_examples", "") or "")
                if not examples_block and hasattr(tool, "get_examples"):
                    try:
                        examples_block = tool.get_examples() or ""
                    except Exception:
                        examples_block = ""
                lines.append(
                    f"- {name}\n"
                    f"  desc: {desc}\n"
                    f"  instructions: {instr}\n"
                    f"  format: {fmt}\n"
                    f"  regex: {rx}\n"
                    f"  forbidden: {forb}\n"
                    f"  caps: {caps}"
                )
                ex_lines = [e.strip() for e in (examples_block or "").splitlines() if e.strip()]
                if ex_lines:
                    lines.append("  examples:")
                    for e in ex_lines[:5]:
                        lines.append(f"    - {e}")
            except Exception as e:
                lines.append(f"- {name}\n  desc: (error building catalogue entry: {e})")
        return "\n".join(lines) if lines else "(none)"
