from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Any

from ..llm_adapters.llm_base import LLMBase
from .tools.agent_tool import AgentTool

logger = logging.getLogger(__name__)

@dataclass
class RefinerPolicy:
    enforce_regex: str = "soft"
    preserve_quoted_phrases: bool = True
    repair_attempts: int = 1
    require_bare_number_match: bool = False 
    external_validator: Optional[Callable[[str], bool]] = None


class ToolQueryRefiner:
    def __init__(self, llm: LLMBase, max_chars: int = 300):
        self.llm = llm
        self.max_chars = int(max_chars)
        logger.info("[ToolQueryRefiner.__init__] Initialized with max_chars=%d", self.max_chars)

    def refine(
        self,
        *,
        user_question: str,
        step_task: str,
        tool: AgentTool,
        policy: Optional[RefinerPolicy] = None,
        previous_step_results: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        tool_display_name = getattr(tool, "name", tool.__class__.__name__)
        tool_display_name = tool_display_name() if callable(tool_display_name) else tool_display_name

        logger.info(
            "[refine] Start refine tool=%s step_task=%r user_question_len=%d",
            tool_display_name,
            step_task,
            len(user_question or ""),
        )

        tool_name_attr = getattr(tool, "name", tool.__class__.__name__)
        tool_name = tool_name_attr() if callable(tool_name_attr) else tool_name_attr

        tool_desc_attr = getattr(tool, "description", "") or getattr(tool, "__doc__", "")
        tool_desc = tool_desc_attr() if callable(tool_desc_attr) else tool_desc_attr

        tool_instr_attr = getattr(tool, "refiner_instructions", "")
        tool_instr = tool_instr_attr() if callable(tool_instr_attr) else (tool_instr_attr or "")

        tool_fmt_attr = getattr(tool, "refiner_input_format", "")
        tool_fmt = tool_fmt_attr() if callable(tool_fmt_attr) else (tool_fmt_attr or "")

        tool_regex_attr = getattr(tool, "refiner_input_regex", None)
        tool_regex = tool_regex_attr() if callable(tool_regex_attr) else tool_regex_attr

        tool_forbid_attr = getattr(tool, "refiner_forbidden", "")
        tool_forbid = tool_forbid_attr() if callable(tool_forbid_attr) else (tool_forbid_attr or "")

        tool_examples_attr = getattr(tool, "refiner_examples", "")
        tool_examples_extra = tool_examples_attr() if callable(tool_examples_attr) else (tool_examples_attr or "")

        tool_policy_attr = getattr(tool, "refiner_policy", None)
        tool_policy = tool_policy_attr() if callable(tool_policy_attr) else tool_policy_attr
        policy = tool_policy or policy or RefinerPolicy()

        examples = ""
        if hasattr(tool, "get_examples"):
            try:
                ge = tool.get_examples() or ""
                examples = (
                    tool_examples_extra
                    + ("\n" if tool_examples_extra and ge else "")
                    + ge
                ).strip()
            except Exception as e:
                examples = tool_examples_extra or ""
                logger.warning("[refine] get_examples failed; using refiner_examples only. error=%r", e)

        hard_cap = self._effective_cap(tool)
        dependency_block = self._build_dependency_block(previous_step_results)

        if str(tool_name).lower() == "calculator":
            expr = (step_task or "").strip()
            if self._looks_like_math_expr(expr):
                patched = expr.replace("^", "**")
                out = self._sanitize(patched)[:hard_cap]
                logger.info("[refine] calculator fast-path: %r -> %r", expr, out)
                return out

        json_header = f'{{ "tool_input": "<string, <={hard_cap} chars>", "reason": "<short>" }}'

        global_guidance = (
            "Global rules:\n"
            "1) Use USER QUESTION as the source of truth for facts and numbers.\n"
            "2) Use DEPENDENT STEP RESULTS as additional facts (when relevant).\n"
            "3) Do not invent facts. If something is missing, write the best tool_input you can from available context.\n"
            "4) Produce a NORMAL tool input string (not URL parameters).\n"
            "5) Obey the TOOL INSTRUCTIONS / INPUT FORMAT / REGEX.\n"
            "6) Output STRICT JSON only.\n"
        )

        prompt = (
            "You convert (USER QUESTION + CURRENT STEP TASK) into the exact INPUT STRING required by one TOOL.\n"
            f"Return STRICT JSON only: {json_header}\n\n"
            f"{global_guidance}\n"
            "TOOL CONTRACT:\n"
            f"Name: {tool_name}\n"
            f"Description: {tool_desc}\n"
            f"Instructions: {tool_instr}\n"
            f"Input format: {tool_fmt}\n"
            f"Input regex (optional): {tool_regex or ''}\n"
            f"Forbidden (optional): {tool_forbid}\n"
            f"Examples (optional):\n{examples}\n\n"
            "CONTEXT:\n"
            f"USER QUESTION: {user_question!r}\n"
            f"CURRENT STEP TASK: {step_task!r}\n\n"
            "DEPENDENT STEP RESULTS (if any):\n"
            f"{dependency_block}\n\n"
            "JSON:"
        )

        logger.debug("[refine] prompt_len=%d", len(prompt))
        logger.debug("[refine] prompt_preview=%r", prompt[:800])

        try:
            raw = (self.llm.generate(prompt) or "").strip()
            logger.debug("[refine] raw_llm_output_len=%d", len(raw))
            logger.debug("[refine] raw_llm_output_preview=%r", raw[:800])

            data = self._safe_parse_json(raw)
            reason = (data.get("reason") or "").strip()
            if reason:
                logger.info("[refine] llm_reason=%r", reason)

            tool_input = self._sanitize((data.get("tool_input") or "").strip())
            logger.info("[refine] candidate_tool_input=%r", tool_input)

            if not tool_input:
                logger.warning("[refine] empty tool_input -> fallback")
                return self._fallback(step_task or user_question)

            if policy.preserve_quoted_phrases:
                tool_input = self._ensure_quoted_phrases(user_question, tool_input)

            bare_num_ok = True
            if policy.require_bare_number_match:
                bare_num_ok = (not self._is_bare_number(tool_input)) or self._is_bare_number(step_task)

            regex_ok = True
            if tool_regex:
                try:
                    regex_ok = bool(re.fullmatch(tool_regex, tool_input, flags=re.IGNORECASE))
                    logger.debug("[refine] regex_check pattern=%r ok=%s", tool_regex, regex_ok)
                except re.error as rex:
                    logger.error("[refine] invalid tool_regex supplied by tool error=%r", rex)
                    regex_ok = True 

            ext_ok = True
            if policy.external_validator:
                try:
                    ext_ok = bool(policy.external_validator(tool_input))
                except Exception as e:
                    logger.warning("[refine] external_validator raised %r; ignoring.", e)
                    ext_ok = True

            needs_repair = (not bare_num_ok) or (tool_regex and not regex_ok) or (not ext_ok)
            if policy.repair_attempts > 0 and needs_repair:
                for attempt in range(policy.repair_attempts):
                    logger.info("[refine] attempting repair pass %d", attempt + 1)
                    repaired = self._repair_candidate(
                        user_question=user_question,
                        step_task=step_task,
                        candidate=tool_input,
                        tool_name=str(tool_name),
                        tool_desc=str(tool_desc or ""),
                        tool_instr=str(tool_instr or ""),
                        tool_fmt=str(tool_fmt or ""),
                        tool_regex=tool_regex,
                    )
                    if repaired and repaired != tool_input:
                        tool_input = repaired
                        logger.info("[refine] repair produced new candidate=%r", tool_input)

                        # re-check
                        if policy.require_bare_number_match:
                            bare_num_ok = (not self._is_bare_number(tool_input)) or self._is_bare_number(step_task)
                        if tool_regex:
                            try:
                                regex_ok = bool(re.fullmatch(tool_regex, tool_input, flags=re.IGNORECASE))
                            except re.error:
                                regex_ok = True
                        if policy.external_validator:
                            try:
                                ext_ok = bool(policy.external_validator(tool_input))
                            except Exception:
                                ext_ok = True
                    break  

            if policy.require_bare_number_match and not bare_num_ok:
                logger.warning("[refine] bare-number mismatch; proceeding (lenient).")

            if tool_regex and not regex_ok:
                msg = "[refine] tool_regex mismatch."
                if policy.enforce_regex == "hard":
                    logger.warning(msg + " -> fallback")
                    return self._fallback(step_task or user_question)
                elif policy.enforce_regex == "soft":
                    logger.warning(msg + " Proceeding due to soft policy.")
                else:
                    logger.debug(msg + " Ignored due to policy off.")

            if not ext_ok:
                logger.warning("[refine] external_validator failed; proceeding (lenient).")

            final_out = tool_input[:hard_cap]
            if len(tool_input) > hard_cap:
                logger.debug("[refine] truncating tool_input to hard_cap=%d", hard_cap)

            logger.info("[refine] final_tool_input=%r", final_out)
            return final_out

        except Exception as e:
            logger.exception("[refine] exception -> fallback error=%r", e)
            return self._fallback(step_task or user_question)


    def _build_dependency_block(self, previous_step_results: Optional[List[Dict[str, Any]]]) -> str:
        if not previous_step_results:
            return "(none)"

        lines: List[str] = []
        for item in previous_step_results:
            try:
                idx = item.get("index") or item.get("step_index") or ""
                instr = (item.get("instruction") or "").strip()

                facts = item.get("facts", None)
                if isinstance(facts, list) and facts:
                    lines.append(f"- Step {idx}: {instr}")
                    for f in facts:
                        try:
                            k = (f.get("key") or "").strip()
                            v = (f.get("value") or "").strip()
                            n = f.get("number", None)
                            u = (f.get("unit") or "").strip()
                            if n is not None and (not v):
                                lines.append(f"  fact: {k} = {n}{(' ' + u) if u else ''}".strip())
                            else:
                                lines.append(f"  fact: {k} = {v}{(' (' + u + ')') if u else ''}".strip())
                        except Exception as e:
                            logger.warning("[_build_dependency_block] failed to render fact %r error=%r", f, e)
                    continue

                obs = (item.get("observation") or "").strip()
                evid = (item.get("evidence") or item.get("evidence_snippet") or "").strip()
                raw_res = (item.get("raw_result") or "").strip()

                lines.append(f"- Step {idx}: {instr}")
                if obs:
                    lines.append(f"  observation: {obs}")
                if evid:
                    lines.append(f"  evidence: {evid}")
                if raw_res:
                    snippet = raw_res[:500].strip()
                    lines.append(f"  raw_result_snippet: {snippet}")

            except Exception as e:
                logger.warning("[_build_dependency_block] failed to render previous step %r error=%r", item, e)

        return "\n".join(lines) if lines else "(none)"

    def _effective_cap(self, tool: AgentTool) -> int:
        cap_attr = getattr(tool, "refiner_max_chars", None)
        cap = cap_attr() if callable(cap_attr) else cap_attr
        try:
            cap_int = int(cap) if cap is not None else self.max_chars
        except Exception as e:
            logger.warning("[effective_cap] invalid cap on tool, using default. error=%r", e)
            cap_int = self.max_chars
        out = max(1, min(self.max_chars, cap_int))
        logger.debug("[effective_cap] cap_attr=%r resolved_cap=%d", cap, out)
        return out

    def _fallback(self, text: str) -> str:
        out = self._sanitize(text)[: self.max_chars]
        logger.info("[fallback] out=%r", out)
        return out

    def _sanitize(self, s: str) -> str:
        before = s
        s = (s or "").replace("\n", " ").replace("\r", " ").strip()
        s = re.sub(r"\s{2,}", " ", s)
        logger.debug("[sanitize] before=%r after=%r", before, s)
        return s

    def _safe_parse_json(self, raw: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw)
            logger.debug("[_safe_parse_json] parsed via direct json.loads")
            if isinstance(parsed, dict):
                return parsed
            return {"tool_input": ""}
        except Exception as e:
            logger.debug("[_safe_parse_json] direct parse failed error=%r", e)
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(raw[start : end + 1])
                    logger.debug("[_safe_parse_json] parsed via substring braced extraction")
                    if isinstance(parsed, dict):
                        return parsed
                except Exception as e2:
                    logger.warning("[_safe_parse_json] substring parse failed error=%r", e2)
            logger.warning("[_safe_parse_json] returning empty dict fallback")
            return {"tool_input": ""}

    def _ensure_quoted_phrases(self, src: str, cand: str) -> str:
        phrases = re.findall(r'"([^"]+)"', src or "")
        if not phrases:
            return cand
        missing = [p for p in phrases if p not in cand]
        if not missing:
            return cand
        repaired = cand.rstrip()
        for p in missing:
            repaired += ("" if repaired.endswith(" ") else " ") + f'"{p}"'
        logger.debug("[ensure_quoted_phrases] added=%s result=%r", missing, repaired)
        return repaired

    def _repair_candidate(
        self,
        *,
        user_question: str,
        step_task: str,
        candidate: str,
        tool_name: str,
        tool_desc: str,
        tool_instr: str,
        tool_fmt: str,
        tool_regex: Optional[str],
    ) -> Optional[str]:
        repair_prompt = (
            "You will minimally fix a tool input string.\n"
            'Return STRICT JSON only: { "tool_input": "<corrected string>" }\n\n'
            f"TOOL NAME: {tool_name}\n"
            f"DESCRIPTION: {tool_desc}\n"
            f"INSTRUCTIONS: {tool_instr}\n"
            f"INPUT FORMAT: {tool_fmt}\n"
            f"INPUT REGEX (optional): {tool_regex or ''}\n\n"
            f"USER QUESTION: {user_question!r}\n"
            f"CURRENT STEP TASK: {step_task!r}\n"
            f"CURRENT CANDIDATE: {candidate!r}\n\n"
            "Constraints:\n"
            "- Do not add new facts.\n"
            "- Keep the user's intent and the step task.\n"
            "- Make the smallest edits necessary to comply with the tool contract/regex.\n"
            "- Output JSON only.\n"
            "JSON:"
        )
        try:
            raw = (self.llm.generate(repair_prompt) or "").strip()
            data = self._safe_parse_json(raw)
            fixed = self._sanitize((data.get("tool_input") or "").strip())
            return fixed or None
        except Exception as e:
            logger.warning("[repair_candidate] repair failed error=%r", e)
            return None

    def _looks_like_math_expr(self, s: str) -> bool:
        if not s:
            return False
        return bool(re.fullmatch(r"[0-9\s+\-*/%^().,eE]+", s))

    def _is_bare_number(self, s: str) -> bool:
        res = bool(re.fullmatch(r"\s*-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?\s*", s or ""))
        logger.debug("[_is_bare_number] s=%r -> %s", s, res)
        return res
