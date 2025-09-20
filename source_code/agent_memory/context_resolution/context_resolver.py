# agent_memory/context_resolution/llm_context_resolver.py
from __future__ import annotations
import json, re, logging, textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from llm_adapters.llm_base import LLMBase

logger = logging.getLogger(__name__)

_NUM_TOKEN = re.compile(
    r"(?P<num>\d{1,3}(?:[ ,._]\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s*(?P<scale>thousand|million|billion|trillion))?"
    r"(?P<pct>\s*%)?",
    re.IGNORECASE,
)

_DATE_TOKEN = re.compile(
    r"\b("
    r"\d{4}-\d{2}-\d{2}"
    r"|(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4})"
    r"|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4})"
    r")\b",
    re.IGNORECASE,
)

_QUOTED_TOKEN = re.compile(r'(?:"([^"]+)"|\'([^\']+)\')')
_PROPER_NOUN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")

@dataclass
class Candidate:
    id: str
    dtype: str
    value: Union[float, int, str, bool, Dict[str, Any]]
    span: str
    source: str
    unit: Optional[str] = None
    is_percent: bool = False
    iso: Optional[str] = None

class ContextResolver:
    def __init__(
        self,
        llm: LLMBase,
        max_context_characters: int = 2000,
        max_candidates: int = 24,
        context_window_characters: int = 48,
        entity_extractor: Any = None,
        entity_store: Any = None,
        drop_single_char_numeric_tokens: bool = True,
        reject_grounded_text_with_external_numbers: bool = True,
    ):
        self.llm = llm
        self.max_context_characters = int(max_context_characters)
        self.max_candidates = max(2, int(max_candidates))
        self.context_window_characters = max(16, int(context_window_characters))
        self.entity_extractor = entity_extractor
        self.entity_store = entity_store
        self.drop_single_char_numeric_tokens = bool(drop_single_char_numeric_tokens)
        self.reject_grounded_text_with_external_numbers = bool(reject_grounded_text_with_external_numbers)

    def ground(
        self,
        original_query: str,
        current_subquery: str,
        prior_pairs: List[Tuple[str, str]],
        conversation_context: str = "",
        dsl_segments: Optional[List[str]] = None,
        query_id: Optional[str] = None,
        subquery_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        step_text = (current_subquery or "").strip()
        if not step_text:
            return {"grounded_text": current_subquery or "", "calc_expr": ""}

        candidates = self._collect_candidates(prior_pairs, step_text, conversation_context, query_id=query_id, subquery_id=subquery_id)
        if not candidates:
            return {"grounded_text": step_text, "calc_expr": ""}

        if len(candidates) > self.max_candidates:
            candidates = candidates[-self.max_candidates:]

        prompt = self._build_prompt(
            original_query=original_query,
            step_text=step_text,
            candidates=candidates,
            conversation=self._truncate(conversation_context, self.max_context_characters),
            dsl_segments=dsl_segments or [],
        )

        raw = (self.llm.generate(prompt) or "").strip()
        plan = self._parse_json_object(raw)
        if not isinstance(plan, dict):
            return self._need_clarification(step_text, "unparseable")

        ok, reason = self._validate_plan(plan, candidates, dsl_segments or [])
        if not ok:
            if reason == "missing_requirements":
                return {
                    "grounded_text": step_text,
                    "calc_expr": "",
                    "needs": {"type": "fetch", "missing": plan.get("missing") or [], "bindings": plan.get("bindings") or []},
                    "reason": reason,
                }
            return self._need_clarification(step_text, reason)

        compiled = self._compile(plan, {c.id: c for c in candidates})
        grounded_text = plan.get("grounded_text") or self._render_grounded_text(plan, candidates)
        if self.reject_grounded_text_with_external_numbers and self._has_external_numbers(grounded_text, candidates):
            grounded_text = step_text
        return {**compiled, "grounded_text": grounded_text}

    def _build_prompt(
        self,
        original_query: str,
        step_text: str,
        candidates: List[Candidate],
        conversation: str,
        dsl_segments: List[str],
    ) -> str:
        def show_val(c: Candidate) -> str:
            if c.dtype == "number":
                return self._format_number(float(c.value))
            if c.dtype == "date":
                return c.iso or str(c.value)
            if c.dtype in {"string", "entity"}:
                return f'"{str(c.value)}"'
            if c.dtype == "boolean":
                return "true" if c.value else "false"
            return str(c.value)

        candidate_block = "\n".join(
            f'- id:{c.id} type:{c.dtype} value:{show_val(c)} span:"{c.span}" ctx:"{c.source}"'
            for c in candidates
        )
        dsl_block = "\n".join(f"- {s}" for s in dsl_segments) if dsl_segments else "(none)"
        return textwrap.dedent(f"""
        Map STEP to an operation using ONLY CANDIDATES. Do not invent values. Use ids as given. If a required value is absent, list it in "missing" and set intent "none".

        REQUIREMENTS:
        {dsl_block}

        PRIOR CONTEXT:
        {conversation or "(none)"}

        ORIGINAL QUERY:
        {original_query}

        STEP:
        {step_text}

        CANDIDATES:
        {candidate_block}

        INTENTS:
        - "select": choose one candidate value
        - "compare": binary comparison on two operands (==, !=, <, >)
        - "contains": string containment test (A in B)
        - "arithmetic": + - * / on two numeric operands
        - "percent_relative": (A / B) * 100
        - "percent_of": (P / 100) * Base
        - "date_diff_days": days between two dates (A - B)
        - "none": cannot resolve

        STRICT JSON ONLY:
        {{
          "intent": "select" | "compare" | "contains" | "arithmetic" | "percent_relative" | "percent_of" | "date_diff_days" | "none",
          "operation": "+|-|*|/|==|!=|<|>|in|null",
          "operands": ["id:..." , "id:..."],
          "selected": "id:..." | null,
          "bindings": [{{ "requirement": "<string>", "id": "id:..." | null }}],
          "missing": ["<string>"],
          "grounded_text": "<concise rendering>",
          "notes": "<=12 words>"
        }}

        RULES:
        - Use only ids from CANDIDATES.
        - For percent_relative: operands[0]=numerator, operands[1]=denominator.
        - For percent_of: operands[0]=percent value, operands[1]=base value.
        - For contains: operation must be "in" and both operands refer to strings/entities.
        - For date_diff_days: both operands must be dates; result in days.
        - If any needed binding is missing, set intent "none".
        - JSON only.
        """)

    def _validate_plan(self, plan: Dict[str, Any], candidates: List[Candidate], requirements: List[str]) -> Tuple[bool, str]:
        ids = {c.id for c in candidates}
        intent = plan.get("intent")
        if intent not in {"select","compare","contains","arithmetic","percent_relative","percent_of","date_diff_days","none"}:
            return False, "bad_intent"

        if any(plan.get("missing") or []):
            return False, "missing_requirements"

        if requirements:
            b = plan.get("bindings")
            if not isinstance(b, list):
                return False, "bad_bindings"
            seen = [x.get("requirement") for x in b if isinstance(x, dict)]
            if sorted((r or "").strip() for r in seen) != sorted((r or "").strip() for r in requirements):
                return False, "bindings_mismatch"

        if intent == "none":
            return False, "no_intent"

        if intent == "select":
            s = plan.get("selected")
            return (isinstance(s, str) and s in ids, "unknown_id" if not s or s not in ids else "")

        if intent in {"compare","contains","arithmetic","percent_relative","percent_of","date_diff_days"}:
            ops = plan.get("operands")
            if not (isinstance(ops, list) and len(ops) == 2 and all(isinstance(x, str) for x in ops)):
                return False, "operands_fields"
            a, b = ops
            if a not in ids or b not in ids or a == b:
                return False, "bad_operands"
            if intent == "compare":
                if plan.get("operation") not in {"==","!=", "<", ">"}:
                    return False, "bad_operation"
            if intent == "contains":
                if plan.get("operation") != "in":
                    return False, "bad_operation"
            if intent == "arithmetic":
                if plan.get("operation") not in {"+","-","*","/"}:
                    return False, "bad_operation"
            if intent in {"percent_relative","percent_of","date_diff_days"}:
                pass
            return True, ""

        return False, "unexpected"

    def _compile(self, plan: Dict[str, Any], by_id: Dict[str, Candidate]) -> Dict[str, Any]:
        intent = plan.get("intent")
        if intent == "select":
            cid = plan.get("selected")
            c = by_id[cid]
            return {"calc_expr": "", "selection": {"id": cid, "type": c.dtype, "value": c.value, "span": c.span}}

        if intent == "compare":
            a_id, b_id = plan["operands"]
            a, b = by_id[a_id], by_id[b_id]
            return {"calc_expr": "", "comparison": {"op": plan["operation"], "left": {"id": a_id, "type": a.dtype, "value": a.value}, "right": {"id": b_id, "type": b.dtype, "value": b.value}}}

        if intent == "contains":
            a_id, b_id = plan["operands"]
            a, b = by_id[a_id], by_id[b_id]
            return {"calc_expr": "", "contains": {"needle": {"id": a_id, "value": str(a.value)}, "haystack": {"id": b_id, "value": str(b.value)}}}

        if intent == "arithmetic":
            a_id, b_id = plan["operands"]; op = plan["operation"]
            a, b = by_id[a_id], by_id[b_id]
            if b.value == 0 and op == "/":
                return {"calc_expr": ""}
            return {"calc_expr": f"{self._format_number(float(a.value))}{op}{self._format_number(float(b.value))}"}

        if intent == "percent_relative":
            a_id, b_id = plan["operands"]
            a, b = by_id[a_id], by_id[b_id]
            if float(b.value) == 0:
                return {"calc_expr": ""}
            return {"calc_expr": f"({self._format_number(float(a.value))}/{self._format_number(float(b.value))})*100"}

        if intent == "percent_of":
            p_id, base_id = plan["operands"]
            p, base = by_id[p_id], by_id[base_id]
            return {"calc_expr": f"({self._format_number(float(p.value))}/100)*{self._format_number(float(base.value))}"}

        if intent == "date_diff_days":
            a_id, b_id = plan["operands"]
            a, b = by_id[a_id], by_id[b_id]
            if not (a.iso and b.iso):
                return {"calc_expr": ""}
            return {"calc_expr": f'(("{a.iso}")-("{b.iso}"))'}

        return {"calc_expr": ""}

    def _collect_candidates(self, prior_pairs: List[Tuple[str, str]], step_text: str, conversation_context: str, query_id: Optional[str], subquery_id: Optional[str]) -> List[Candidate]:
        out: List[Candidate] = []
        k = 0

        def add(dtype: str, value: Any, span: str, origin: str, unit: Optional[str] = None, is_percent: bool = False, iso: Optional[str] = None):
            nonlocal k, out
            out.append(Candidate(id=f"c{k}", dtype=dtype, value=value, span=span, source=origin, unit=unit, is_percent=is_percent, iso=iso))
            k += 1

        def window(text: str, lo: int, hi: int) -> str:
            ctx_lo = max(0, lo - self.context_window_characters)
            ctx_hi = min(len(text), hi + self.context_window_characters)
            return text[ctx_lo:ctx_hi].replace("\n", " ")

        def harvest(text: str, origin: str):
            if not text:
                return
            t = str(text)

            for m in _NUM_TOKEN.finditer(t):
                num_raw = (m.group("num") or "").replace(",", "").replace("_", "").replace(" ", "")
                if self.drop_single_char_numeric_tokens and len(num_raw) == 1 and not (m.group("scale") or m.group("pct")):
                    continue
                v = self._normalize_number(m)
                if v is not None:
                    add("number", v, m.group(0).strip(), f"{origin}: {window(t, *m.span())}", is_percent=bool(m.group("pct")))

            for m in _DATE_TOKEN.finditer(t):
                iso = self._to_iso(m.group(0))
                if iso:
                    add("date", iso, m.group(0).strip(), f"{origin}: {window(t, *m.span())}", iso=iso)

            for m in _QUOTED_TOKEN.finditer(t):
                s = (m.group(1) or m.group(2) or "").strip()
                if s:
                    add("string", s, m.group(0), f"{origin}: {window(t, *m.span())}")

            if self.entity_extractor:
                try:
                    ent = self.entity_extractor.extract(t, query_id=query_id or "", subquery_id=subquery_id)
                    if getattr(ent, "text", None):
                        add("entity", ent.text, ent.text, f"{origin}: entity")
                except Exception:
                    pass
            else:
                for m in _PROPER_NOUN.finditer(t):
                    token = m.group(1).strip()
                    if token.lower() not in {"i","you","we","he","she","it","they","the","a","an","and","or"}:
                        add("entity", token, token, f"{origin}: {window(t, *m.span())}")

            for token, val in ((" true ", True), (" false ", False)):
                if token in f" {t.lower()} ":
                    add("boolean", val, token.strip(), f"{origin}: boolean")

        def names_from_text(text: str) -> List[str]:
            s = set()
            if self.entity_extractor:
                try:
                    e = self.entity_extractor.extract(text, query_id=query_id or "", subquery_id=subquery_id)
                    if getattr(e, "text", None):
                        s.add(e.text)
                except Exception:
                    pass
            for m in _PROPER_NOUN.finditer(text or ""):
                token = m.group(1).strip()
                if token.lower() not in {"i","you","we","he","she","it","they","the","a","an","and","or"}:
                    s.add(token)
            return list(s)

        def coerce_value(v: Any) -> Tuple[str, Any, Optional[str], bool, Optional[str]]:
            if isinstance(v, bool):
                return "boolean", bool(v), None, False, None
            if isinstance(v, (int, float)):
                return "number", float(v), None, False, None
            if isinstance(v, dict):
                raw = v.get("value")
                unit = v.get("unit")
                if isinstance(raw, (int, float)):
                    return "number", float(raw), unit, False, None
                if isinstance(raw, str):
                    iso = self._to_iso(raw)
                    if iso:
                        return "date", iso, None, False, iso
                    m = _NUM_TOKEN.search(raw)
                    if m:
                        nv = self._normalize_number(m)
                        return "number", nv if nv is not None else raw, unit, bool(m.group("pct")), None
                    return "string", raw, unit, False, None
                return "string", str(raw), unit, False, None
            if isinstance(v, str):
                iso = self._to_iso(v)
                if iso:
                    return "date", iso, None, False, iso
                m = _NUM_TOKEN.search(v)
                if m:
                    nv = self._normalize_number(m)
                    return "number", nv if nv is not None else v, None, bool(m.group("pct")), None
                low = v.strip().lower()
                if low in {"true","false"}:
                    return "boolean", (low == "true"), None, False, None
                return "string", v, None, False, None
            return "string", str(v), None, False, None

        for q, a in prior_pairs:
            harvest(a, "prior_answer")
            harvest(q, "prior_subquery")

        harvest(conversation_context, "conversation_memory")
        harvest(step_text, "current_step")

        if self.entity_store:
            names = names_from_text(step_text)
            for name in names:
                try:
                    ent = self.entity_store.resolve(name)
                except Exception:
                    ent = None
                if not ent:
                    continue
                seen_by_predicate: Dict[str, Any] = {}
                try:
                    all_f = self.entity_store.all_facts(ent.entity_id)
                except Exception:
                    all_f = []
                for f in all_f:
                    key = str(getattr(f, "predicate", "")).strip().lower()
                    if not key:
                        continue
                    seen_by_predicate[key] = f
                for pred, fact in seen_by_predicate.items():
                    dtype, val, unit, is_pct, iso = coerce_value(getattr(fact, "value", None))
                    add(dtype, val, str(getattr(fact, "value", "")), f"entity_store: {name}.{pred}", unit=unit, is_percent=is_pct, iso=iso)
                add("entity", name, name, "entity_store: entity")

        return out

    def _need_clarification(self, step_text: str, reason: str) -> Dict[str, Any]:
        question = f'Which values from earlier context should be used to resolve: “{step_text}”?'
        return {"grounded_text": step_text, "calc_expr": "", "needs": "clarification", "reason": reason, "question": question}

    def _truncate(self, s: str, n: int) -> str:
        s = (s or "").strip()
        return (s[:n] + "…") if len(s) > n else s

    def _format_number(self, v: float) -> str:
        iv = int(v)
        return str(iv) if abs(v - iv) < 1e-9 else f"{v:.12f}".rstrip("0").rstrip(".")

    def _normalize_number(self, m: re.Match) -> Optional[float]:
        num = m.group("num").replace(",", "").replace("_", "").replace(" ", "")
        try:
            val = float(num)
        except Exception:
            return None
        scale = (m.group("scale") or "").lower()
        mult = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}.get(scale, 1.0)
        return val * mult

    def _to_iso(self, s: str) -> Optional[str]:
        s = s.strip()
        fmts = ["%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"]
        for f in fmts:
            try:
                dt = datetime.strptime(s, f)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                continue
        return None

    def _parse_json_object(self, raw: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(raw)
        except Exception:
            pass
        try:
            b, e = raw.find("{"), raw.rfind("}")
            if b != -1 and e != -1 and e > b:
                return json.loads(raw[b:e+1])
        except Exception:
            pass
        return None

    def _render_grounded_text(self, plan: Dict[str, Any], candidates: List[Candidate]) -> str:
        by_id = {c.id: c for c in candidates}
        intent = plan.get("intent")
        if intent == "select":
            s = plan.get("selected")
            return f"Use {by_id[s].span}."
        if intent in {"compare","contains","arithmetic","percent_relative","percent_of","date_diff_days"}:
            a, b = plan.get("operands", [None, None])
            op = plan.get("operation")
            l = by_id.get(a).span if a in by_id else "A"
            r = by_id.get(b).span if b in by_id else "B"
            return f"Compute {l} {op} {r}."
        return "Resolve."

    def _has_external_numbers(self, text: str, candidates: List[Candidate]) -> bool:
        toks = self._extract_numeric_tokens_from_text(text)
        if not toks:
            return False
        cand = self._candidate_numeric_set(candidates)
        return not toks.issubset(cand)

    def _extract_numeric_tokens_from_text(self, s: Optional[str]) -> set:
        if not s:
            return set()
        raw = re.findall(r"\d+(?:\.\d+)?", s)
        out = set()
        for t in raw:
            try:
                v = float(t.replace(",", "").replace("_", ""))
                out.add(self._format_number(v))
            except Exception:
                continue
        return out

    def _candidate_numeric_set(self, candidates: List[Candidate]) -> set:
        out = set()
        for c in candidates:
            if c.dtype == "number":
                try:
                    out.add(self._format_number(float(c.value)))
                except Exception:
                    pass
            span_nums = _NUM_TOKEN.findall(c.span or "")
            for num, scale, pct in span_nums:
                try:
                    fake_match = re.match(r".*", "")
                except Exception:
                    fake_match = None
                m = re.search(_NUM_TOKEN, c.span or "")
                if m:
                    v = self._normalize_number(m)
                    if v is not None:
                        out.add(self._format_number(float(v)))
        return out
