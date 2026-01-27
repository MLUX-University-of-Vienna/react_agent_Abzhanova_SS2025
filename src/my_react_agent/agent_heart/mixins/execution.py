from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ...agent_memory.data_structures import (
    Evidence,
    EvidenceExtract,
    Step,
    StepPlan,
    StepDecision as MemStepDecision,
    Entity,
    Query,
    StepMeta,
    entity_with_evidence,
    new_entity,
    query_upsert_entity,
    step_add_evidence,
    step_set_result,
)

from ...tool_management.tool_query_refiner import RefinerPolicy

from ..logger import logger
from .text_utils import TextUtilsMixin

from ..agent_prompts.prompts_ids import PromptId

class ExecutionMixin(TextUtilsMixin):
    _NUM_RE = re.compile(r"(?<![A-Za-z0-9_])(\d{1,3}(?:[,\s]\d{3})+|\d+(?:\.\d+)?)")

    _FACT_SPLIT_RE = re.compile(r"(?:^|\s)(\d+)\.\s+")

    def _allowed_numbers_from_deps(self, prev_for_refiner: List[Dict[str, Any]]) -> List[str]:
        nums: List[str] = []
        for item in prev_for_refiner or []:
            if not isinstance(item, dict):
                continue
            facts = item.get("facts", []) or []
            if not isinstance(facts, list):
                continue
            for f in facts:
                if not isinstance(f, dict):
                    continue
                n = f.get("number", None)
                if n is None:
                    continue
                token = str(n).replace(",", "").strip()
                if re.fullmatch(r"-?\d+(\.\d+)?([eE][+\-]?\d+)?", token):
                    nums.append(token)

        seen = set()
        out: List[str] = []
        for n in nums:
            if n in seen:
                continue
            seen.add(n)
            out.append(n)
        return out

    def _extract_number_literals_from_expr(self, expr: str) -> List[str]:
        return re.findall(r"(?<!\w)-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?", expr or "")

    def _uses_only_allowed_numbers(self, expr: str, allowed: List[str]) -> bool:
        if not allowed:
            return True
        allowed_set = set(allowed)
        for lit in self._extract_number_literals_from_expr(expr):
            if lit not in allowed_set:
                return False
        return True

    def _normalise_number_token(self, s: str) -> Optional[str]:
        if not s:
            return None
        t = s.strip()
        t = t.replace(" ", "").replace(",", "")
        if not t:
            return None
        if re.fullmatch(r"\d+(\.\d+)?", t):
            return t
        return None

    def _parse_summary_facts(self, summary_text: str, focus: str) -> List[Dict[str, Any]]:
        s = (summary_text or "").strip()
        if not s:
            return []

        parts: List[str] = []
        if self._FACT_SPLIT_RE.search(s):
            chunks = self._FACT_SPLIT_RE.split(s)
            for i in range(2, len(chunks), 2):
                txt = (chunks[i] or "").strip()
                if txt:
                    parts.append(txt)
        else:
            lines = [x.strip() for x in re.split(r"[\n\r]+", s) if x.strip()]
            for ln in lines:
                ln2 = re.sub(r"^[\-\•\*\u2022]\s*", "", ln).strip()
                if ln2:
                    parts.append(ln2)

        parts2: List[str] = []
        for p in parts:
            p2 = self._sanitise_to_text(p).strip()
            if p2:
                parts2.append(p2)
            if len(parts2) >= 5:
                break

        facts: List[Dict[str, Any]] = []
        for i, p in enumerate(parts2, start=1):
            num_val = None
            unit_val = ""
            m = self._NUM_RE.search(p)
            if m:
                tok = self._normalise_number_token(m.group(1))
                if tok:
                    num_val = tok
                    tail = p[m.end() :].strip()
                    unit_m = re.match(r"^([A-Za-z%]+)", tail)
                    if unit_m:
                        unit_val = unit_m.group(1)

            facts.append(
                {
                    "key": f"{focus} (fact {i})",
                    "value": p,
                    "number": num_val,
                    "unit": unit_val,
                    "provenance": {"tool": "step_summary", "focus": focus},
                }
            )
        return facts

    def _build_step_summary_evidence_and_facts(self, step: Step) -> Tuple[Step, List[Dict[str, Any]]]:
        try:
            if step is None:
                return step, []

            try:
                if step.result is not None:
                    ef = getattr(step.result, "extracted_facts", None)
                    if isinstance(ef, (list, tuple)) and len(ef) > 0:
                        existing: List[Dict[str, Any]] = [x for x in ef if isinstance(x, dict)]
                        return step, existing
            except Exception:
                pass

            focus = self._sanitise_to_text(step.plan.instruction or "") or "step"

            try:
                if step.evidence and getattr(step.evidence[-1], "tool", "") == "step_summary":
                    payload = None
                    try:
                        payload = getattr(step.evidence[-1], "extracted", None)
                        payload = getattr(payload, "payload", None) if payload is not None else None
                    except Exception:
                        payload = None
                    facts_from_payload: List[Dict[str, Any]] = []
                    if isinstance(payload, dict):
                        fs = payload.get("facts", [])
                        if isinstance(fs, list):
                            facts_from_payload = [x for x in fs if isinstance(x, dict)]
                    return step, facts_from_payload
            except Exception:
                pass

            summary_text = ""

            try:
                if step.result is not None and getattr(step.result, "meta", None) is not None:
                    if (getattr(step.result.meta, "action", "") or "").upper() == "USE_TOOL":
                        summary_text = self._sanitise_to_text(step.result.observation or "")
            except Exception:
                summary_text = ""

            if not summary_text:
                base_ev: Optional[Evidence] = None
                if step.evidence:
                    try:
                        base_ev = step.evidence[-1]
                    except Exception:
                        base_ev = None

                if base_ev is None:
                    obs = ""
                    try:
                        obs = (step.result.observation if step.result else "") or ""
                    except Exception:
                        obs = ""
                    base_ev = Evidence(
                        tool="step_observation",
                        content=obs,
                        url=None,
                        extracted=EvidenceExtract(fields={}, payload=None),
                        as_of=datetime.utcnow(),
                        confidence=(step.result.confidence if step.result else 0.6),
                    )

                summary_text = self._summarise_observation(focus, base_ev)

            facts_list = self._parse_summary_facts(summary_text, focus)

            summary_ev = Evidence(
                tool="step_summary",
                content=summary_text,
                url=None,
                extracted=EvidenceExtract(fields={"focus": focus}, payload={"facts": facts_list}),
                as_of=datetime.utcnow(),
                confidence=(step.result.confidence if step.result else 0.6),
            )

            step2 = step_add_evidence(step, summary_ev)

            if step2.result is not None:
                try:
                    sr2 = replace(step2.result, extracted_facts=tuple(facts_list))
                    step2 = step_set_result(step2, sr2)
                except Exception:
                    pass

            return step2, facts_list
        except Exception:
            return step, []

    def _handle_low_confidence_step_result(
        self,
        *,
        question: str,
        plan: StepPlan,
        decision: MemStepDecision,
        raw_text: str,
        prior_unsuccessful_attempts: Optional[List[Dict[str, Any]]] = None,
    ) -> MemStepDecision:
        tool_catalog = ""
        try:
            tool_catalog = self._tool_catalog_text()
        except Exception:
            tool_catalog = "(none)"

        actions_catalog = self._actions_catalog_text(self.low_conf_actions)
        allowed_names = [a.name for a in self.low_conf_actions]
        allowed_names_str = ", ".join(allowed_names)

        schema = (
            '{"action":'
            f'"one of: {allowed_names_str}",'
            '"tool_name":"<tool name or empty>",'
            '"tool_input":"<tool input or empty>",'
            '"message":"<short reason>"}'
        )

        action_name = (decision.action or "")
        tool_name = (decision.tool_name or "")
        refined_input = (decision.tool_input or plan.instruction or "")

        tool_name_for_prompt = self._sanitise_to_text(tool_name)
        if not tool_name_for_prompt:
            tool_name_for_prompt = self._sanitise_to_text(action_name)

        prompt = self._render_prompt(
            PromptId.LOW_CONF_TOOL_RESULT,
            actions_catalog=actions_catalog,
            tool_catalog=tool_catalog,
            allowed_actions=allowed_names_str,
            step_task=plan.instruction,
            prev_tool_name=tool_name_for_prompt,
            prev_tool_input=refined_input,
            raw_text=raw_text,
            schema=schema,
        )

        logger.info("[ReActAgent._handle_low_confidence_action_result] Prompt length=%d", len(prompt))
        logger.info("[ReActAgent._handle_low_confidence_action_result] Prompt:\n%s", prompt)

        raw_decision = (self.planner_llm.generate(prompt) or "").strip()
        logger.info("[ReActAgent._handle_low_confidence_action_result] RAW: %s", raw_decision or "")

        data = self._safe_parse_json_object(raw_decision)

        action = self._sanitise_to_text(data.get("action", "")).upper()
        new_tool_name = self._sanitise_to_text(data.get("tool_name", ""))

        step_task_raw = data.get("step's_task", "")
        step_task = self._sanitise_to_text(step_task_raw)
        new_tool_input = self._sanitise_to_text(data.get("tool_input", ""))
        if step_task:
            new_tool_input = step_task

        message = self._sanitise_to_text(data.get("message", ""))

        allowed_set = set(a.upper() for a in allowed_names)
        if action not in allowed_set:
            fallback = "ANSWER_BY_ITSELF" if "ANSWER_BY_ITSELF" in allowed_set else next(iter(allowed_set))
            logger.info(
                "[ReActAgent._handle_low_confidence_action_result] Invalid action '%s'; falling back to '%s'",
                action,
                fallback,
            )
            action = fallback

        tried_triples: set[Tuple[str, str, str]] = set()
        try:
            for att in prior_unsuccessful_attempts or []:
                if not isinstance(att, dict):
                    continue
                a0 = self._sanitise_to_text(att.get("action", "")).upper().strip()
                tn0 = self._sanitise_to_text(att.get("tool_name", "")).strip()
                ti0 = self._sanitise_to_text(att.get("tool_input", "")).strip()
                tried_triples.add((a0, tn0, ti0))
        except Exception:
            tried_triples = set()

        triple = (
            self._sanitise_to_text(action).upper().strip(),
            self._sanitise_to_text(new_tool_name).strip(),
            self._sanitise_to_text(new_tool_input).strip(),
        )
        if triple in tried_triples:
            fallback = "ANSWER_BY_ITSELF" if "ANSWER_BY_ITSELF" in allowed_set else next(iter(allowed_set))
            action = fallback
            new_tool_name = ""
            new_tool_input = ""

        logger.info(
            "[ReActAgent._handle_low_confidence_action_result] Parsed action=%s new_tool=%s",
            action,
            new_tool_name,
        )

        return MemStepDecision(
            action=action,
            think="Decision after low-confidence tool result.",
            tool_name=new_tool_name,
            tool_input=new_tool_input,
            message=message,
        )

    def _heuristic_facts_from_step_record(self, r: Dict[str, Any]) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        if not isinstance(r, dict):
            return facts

        instruction = self._sanitise_to_text(r.get("instruction", "")).strip()
        observation = self._sanitise_to_text(r.get("observation", "")).strip()
        evidence = self._sanitise_to_text(r.get("evidence", "")).strip()
        text = (observation or evidence).strip()
        if not text:
            return facts

        nums = []
        for m in self._NUM_RE.finditer(text):
            token = self._normalise_number_token(m.group(1))
            if token:
                nums.append(token)

        seen = set()
        nums2 = []
        for n in nums:
            if n in seen:
                continue
            seen.add(n)
            nums2.append(n)

        if not nums2:
            return facts

        key_base = instruction or "value"
        primary = nums2[0]
        facts.append(
            {
                "key": key_base,
                "value": primary,
                "number": primary,
                "unit": "",
                "provenance": {"tool": "heuristic", "focus": instruction or key_base},
            }
        )
        for extra in nums2[1:3]:
            facts.append(
                {
                    "key": f"{key_base} (alt)",
                    "value": extra,
                    "number": extra,
                    "unit": "",
                    "provenance": {"tool": "heuristic", "focus": instruction or key_base},
                }
            )
        return facts

    def _execute_and_observe(
        self,
        tool_name: str,
        tool_input: str,
        user_question: str,
        *,
        subquestion: Optional[str] = None,
        policy: Optional[RefinerPolicy] = None,
    ) -> Tuple[str, str, str, List[Dict[str, Any]]]:
        effective_question = self._sanitise_to_text(subquestion or user_question)

        if tool_name not in self.tools:
            logger.warning("[ReActAgent._execute_and_observe] Unknown tool=%s, fallback to first tool", tool_name)
            tool_name = next(iter(self.tools.keys()))

        current_tool_name = tool_name
        current_input_seed = self._sanitise_to_text(tool_input) or self._sanitise_to_text(effective_question)

        last_assessment: Optional[Any] = None
        last_extracted_facts: List[Dict[str, Any]] = []

        if current_tool_name not in self.tools:
            logger.warning(
                "[ReActAgent._execute_and_observe] Current tool '%s' not available; falling back to first tool",
                current_tool_name,
            )
            current_tool_name = next(iter(self.tools.keys()))
        tool = self.tools[current_tool_name]

        all_prev: List[Dict[str, Any]] = getattr(self, "_previous_step_results", [])
        active_deps: List[int] = getattr(self, "_active_step_dependencies", []) or []

        if active_deps and all_prev:
            base_prev = [r for r in all_prev if isinstance(r, dict) and r.get("index") in active_deps]
        else:
            base_prev = []

        prev_for_refiner: List[Dict[str, Any]] = []

        for r in base_prev:
            try:
                step_idx = int(r.get("index"))
            except Exception:
                continue

            step_facts: List[Dict[str, Any]] = []
            try:
                ef = r.get("extracted_facts", []) or []
                if isinstance(ef, list):
                    step_facts = [x for x in ef if isinstance(x, dict)]
            except Exception:
                step_facts = []

            if not step_facts:
                try:
                    step_facts = self._heuristic_facts_from_step_record(r) or []
                except Exception:
                    step_facts = []

            prev_for_refiner.append({"index": step_idx, "instruction": r.get("instruction", ""), "facts": step_facts})

        if prev_for_refiner:
            logger.info(
                "[ReActAgent._execute_and_observe] Passing %d previous step result(s) to ToolQueryRefiner (deps=%s).",
                len(prev_for_refiner),
                active_deps or "ALL",
            )

        raw_candidate = current_input_seed
        try:
            refined_candidate = self.tool_query_refiner.refine(
                user_question=user_question,
                step_task=raw_candidate,
                tool=tool,
                policy=policy,
                previous_step_results=prev_for_refiner,
            )
        except Exception as e:
            logger.warning("[ReActAgent._execute_and_observe] Refiner failed error=%r; using raw input", e)
            refined_candidate = raw_candidate

        refined_input = self._sanitise_to_text(refined_candidate) or current_input_seed

        logger.info(
            "[ReActAgent._execute_and_observe] Executing tool=%s input_len=%d",
            current_tool_name,
            len(refined_input or ""),
        )

        tool_failed = False
        result: Optional[Evidence] = None
        raw_text = ""

        try:
            result = self.tool_executor.execute(current_tool_name, refined_input)
        except Exception as e:
            tool_failed = True
            logger.warning("[ReActAgent._execute_and_observe] Tool executor raised error=%r", e)
            raw_text = f"The tool {current_tool_name} failed while trying to answer the question."
        else:
            raw_text = (result.content if result else "") or ""
            last_extracted_facts = []

        if not tool_failed and result is not None:
            selected_info = ""
            try:
                selected_info = self._sanitise_to_text(getattr(last_assessment, "supporting_text", ""))
            except Exception:
                selected_info = ""

            if not selected_info:
                selected_info = self._extract_evidence_snippet(result)

            short_obs = self._summarise_observation(effective_question, result)
            evidence = selected_info

            logger.info(
                "[ReActAgent._execute_and_observe] Observation_len=%d evidence_len=%d raw_len=%d",
                len(short_obs or ""),
                len(evidence or ""),
                len(raw_text or ""),
            )
            return short_obs, evidence, raw_text, (last_extracted_facts or [])

        obs = self._sanitise_to_text(raw_text) or "No immediately relevant information found."
        return obs, "", raw_text, (last_extracted_facts or [])

    def _summarise_observation(self, user_question: str, result: Evidence) -> str:
        raw_text = self._sanitise_to_text(result.content or "")
        prompt = self._render_prompt(
            PromptId.SUMMARISE_OBSERVATION,
            question=user_question,
            tool_result=raw_text,
        )
        out = (self.summariser_llm.generate(prompt) or "").strip()
        return self._sanitise_to_text(out) or "No immediately relevant information found."

    def _extract_evidence_snippet(self, result: Evidence) -> str:
        text = result.content or ""
        if text and len(text) > self.config.observation_snippet_limit:
            return text[: self.config.observation_snippet_limit]
        return text or ""

    def _update_fact_store_from_result(
        self,
        raw_text: str,
        question: str,
        *,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        text = raw_text or ""
        focus = None
        if provenance:
            focus = provenance.get("focus") or provenance.get("entity_text")
        focus = self._sanitise_to_text(focus)
        focus_or_question = focus or question

        schema = (
            '{"facts":[{"key":"<short>",'
            '"value":"<string>",'
            '"number":"<optional number or null>",'
            '"unit":"<optional>"}]}'
        )
        prompt = self._render_prompt(
            PromptId.UPDATE_FACT_STORE,
            N=self.config.fact_schema_max_items,
            focus=focus_or_question,
            question=question,
            text=text,
            schema=schema,
        )
        raw = (self.summariser_llm.generate(prompt) or "").strip()
        data = self._safe_parse_json_object(raw)
        facts = data.get("facts", [])

        facts_list: List[Dict[str, Any]] = []
        if isinstance(facts, list):
            for f in facts:
                if isinstance(f, dict):
                    facts_list.append(f)

        entity_name = ""
        try:
            extractor = getattr(self, "entity_extractor", None)
            if extractor is not None:
                for method_name in ("extract", "extract_entities", "find_entities", "run"):
                    fn = getattr(extractor, method_name, None)
                    if callable(fn):
                        out = fn(focus_or_question)
                        if isinstance(out, list) and out:
                            for it in out:
                                if isinstance(it, str) and it.strip():
                                    entity_name = it.strip()
                                    break
                                if isinstance(it, dict):
                                    nm = it.get("canonical_name") or it.get("name") or it.get("text")
                                    if isinstance(nm, str) and nm.strip():
                                        entity_name = nm.strip()
                                        break
                        if entity_name:
                            break
                        if isinstance(out, dict):
                            ents = out.get("entities")
                            if isinstance(ents, list) and ents:
                                for it in ents:
                                    if isinstance(it, str) and it.strip():
                                        entity_name = it.strip()
                                        break
                                    if isinstance(it, dict):
                                        nm = it.get("canonical_name") or it.get("name") or it.get("text")
                                        if isinstance(nm, str) and nm.strip():
                                            entity_name = nm.strip()
                                            break
                        if entity_name:
                            break
        except Exception:
            entity_name = ""

        if not entity_name:
            entity_name = focus_or_question.strip() or "global"

        try:
            q = self.query_memory.current_query_model
            if not q:
                return facts_list

            existing: Optional[Entity] = None
            for e in (q.entities or ()):
                if (e.canonical_name or "").strip().lower() == entity_name.strip().lower():
                    existing = e
                    break

            ent = existing or new_entity(entity_name)

            ev = Evidence(
                tool=str((provenance or {}).get("tool") or "unknown"),
                content=text,
                url=None,
                extracted=EvidenceExtract(
                    fields={"focus": focus_or_question},
                    payload={"facts": facts_list} if isinstance(facts_list, list) else {"facts": []},
                ),
                as_of=datetime.utcnow(),
                confidence=0.6,
            )
            ent = entity_with_evidence(ent, ev)

            q2 = query_upsert_entity(q, ent)
            self.query_memory.current_query_model = q2
        except Exception:
            pass

        return facts_list
