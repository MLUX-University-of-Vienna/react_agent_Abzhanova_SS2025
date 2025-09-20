import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from llm_adapters.llm_base import LLMBase
from .query_separation_strategy import QuerySeparationStrategy


def _json_extract_first_object(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    b, e = s.find("{"), s.rfind("}")
    if b != -1 and e != -1 and e > b:
        try:
            return json.loads(s[b:e + 1])
        except Exception:
            return None
    return None


logger = logging.getLogger(__name__)


class LLMLogicalSeparationStrategy(QuerySeparationStrategy):
    """
    Splits a complex query by first emitting a compact logic DSL, then converts that
    (together with the original query) into a minimal DAG plan (JSON). The final
    output is a list of ordered, actionable subqueries (step texts).

    Modifications:
    - Tighten planner prompt with deterministic variable-naming derived from the DSL.
    - Require a resolution step for indirect entities (relation-defined entities).
    - Validate and, when safe, auto-correct output variable names so they preserve
      all entity modifiers from the DSL (prevents dropping role words like "capital_city").
    """

    def __init__(self, llm: LLMBase):
        self.llm = llm

    # ---------- public ----------

    def separate(self, query: str) -> List[str]:
        query = (query or "").strip()
        if not query:
            return []

        # 1) Ask for DSL
        dsl_prompt = (
            "SYSTEM: Convert the user's question into a logical representation using the following DSL:\n"
            "- Use field(entity) expressions, joined by logical operators: ∧ (and), ∨ (or), ¬ (not)\n"
            "- Be strict: only represent what is explicitly asked.\n"
            "- Do not infer or expand the question.\n"
            f"User query: '{query}'\n"
            "Output only DSL expressions — no bullets, markdown, or explanation:"
        )
        logger.debug("Separation prompt:%s", dsl_prompt)
        logic = (self.llm.generate(dsl_prompt) or "").strip()
        logger.debug("Received logic DSL: %s", logic)

        # Split DSL on ∧ (and). Keep order.
        segments = [seg.strip() for seg in logic.split('∧') if seg.strip()] if logic else []

        # 2) Build planner prompt (JSON) using both the query and (if present) the DSL
        plan_prompt = self._make_plan_prompt(query)
        if segments:
            dsl_block = "\n".join(segments)
            # Inject the DSL just before the terminal "JSON ONLY:" marker.
            plan_prompt = plan_prompt.replace(
                "JSON ONLY:",
                "Use the following DSL decomposition as authoritative; do not introduce facts not entailed by it.\n"
                f"DSL:\n{dsl_block}\n\nJSON ONLY:"
            )

        logger.debug("Planner prompt:\n%s", plan_prompt)

        raw_plan = (self.llm.generate(plan_prompt) or "").strip()
        logger.debug("[hybrid] Planner JSON raw:\n%s", raw_plan)

        plan = _json_extract_first_object(raw_plan) or {}

        # Validate and, if possible, auto-correct variable names to match DSL semantics
        try:
            plan, renamed = self._validate_and_fix_plan(plan, segments)
            if renamed:
                logger.debug("Planner outputs auto-corrected to canonical names: %s", renamed)
        except Exception as e:
            logger.debug("Plan validation/auto-fix skipped due to error: %s", e)

        steps = plan.get("steps")
        if isinstance(steps, list) and steps:
            out = [self._sanitize_text(str(s.get("text", "")).strip()) for s in steps if str(s.get("text", "")).strip()]
            if out:
                logger.debug("Subqueries (from planner JSON): %s", out)
                return out

        # 3) Fallbacks
        if segments:
            # emit DSL lines as minimal steps (very light fallback)
            fallback = [self._sanitize_text(seg) for seg in segments]
            logger.debug("Fallback subqueries (from DSL lines): %s", fallback)
            return fallback

        # Last resort: return the whole query as one step
        logger.warning("No valid plan produced; returning the original query.")
        return [query]

    # ---------- planner prompt (tightened) ----------

    def _make_plan_prompt(self, query: str) -> str:
        return (
            "You are a planner. Read the USER QUERY and return the smallest set of unambiguous, executable steps; "
            "write each step's `text` as a plain-English instruction that names ENTITIES, not variables.\n"
            "Do NOT add a final 'output the result' step.\n"
            "Return STRICT JSON only with this schema:\n"
            "{\n"
            '  "steps": [\n'
            '    {"id": "s1", "text": "<one self-contained natural-language step>", "depends_on": ["<ids>"], '
            '     "outputs": {"<var>": "<what it represents (verbal, no formulas)>"}}\n'
            "  ]\n"
            "}\n"
            "\n"
            "Hard rules (follow ALL):\n"
            "- JSON only; no markdown or commentary outside JSON.\n"
            "- ***ABSOLUTE BAN in `text`:***\n"
            "  • any variable identifiers (especially snake_case like foo_bar, or any exact output variable names)\n"
            "  • math operators or equations (+ - * / % ** // ^ =)\n"
            "  • function/DSL call patterns like population(USA) or field(Entity)\n"
            "  Regex guards that MUST NOT match inside `text`:\n"
            "    • [()+\\-*/=%^]\n"
            "    • [A-Za-z_][A-Za-z0-9_]*\\s*\\(.*\\)\n"
            "    • \\b[a-z]+(?:_[a-z0-9]+){1,}\\b   (disallow snake_case tokens)\n"
            "- In `text`, refer to concepts verbally only, e.g. "
            '"Calculate the percentage of the population of the capital city of Austria relative to the population of the United States." '
            "NOT: 'Calculate the percentage of austria_population relative to usa_population.'\n"
            "- `outputs`:\n"
            "  • Keys are variables in snake_case.\n"
            "  • Values are short verbal descriptions (no formulas, no function calls, no variables).\n"
            "- If an entity is defined by a relation (e.g., 'capital city of Austria'), include an early step that resolves it verbally; later steps may reference that entity verbally (not by variable name).\n"
            "- Dependencies must be acyclic and reference only earlier step ids (never the current id). "
            "Do not include a step's own id in its `depends_on`.\n"
            "- If a step conceptually uses results of prior steps, include those step ids in `depends_on`, but do not mention ids or variables in `text`.\n"
            "- Use as few steps as possible while remaining clear.\n"
            "\n"
            f"USER QUERY:\n{query}\n\n"
            "JSON ONLY:"
        )




    # ---------- validation & fixing ----------

    def _validate_and_fix_plan(self, plan: Dict[str, Any], dsl_segments: List[str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Ensure outputs preserve entity modifiers by enforcing the canonical
        <entity>_<field> snake_case naming derived from DSL field(entity) nodes.

        Returns: (possibly-updated plan, {old_name: new_name} of changes applied)
        """
        steps = plan.get("steps")
        if not isinstance(steps, list):
            return plan, {}

        # Extract field(entity) pairs from DSL segments
        dsl_pairs = self._dsl_pairs_from_segments(dsl_segments)
        # Map normalized expression -> canonical var name
        expr_to_canonical: Dict[str, str] = {}
        for field, entity in dsl_pairs:
            canonical = f"{self._to_snake(entity)}_{self._to_snake(field)}"
            expr_key = f"{field.lower()}({entity.lower()})"
            expr_to_canonical[expr_key] = canonical

        renamed: Dict[str, str] = {}

        # Pass 1: rename outputs within each step to canonical names where we can determine them
        for step in steps:
            outputs = step.get("outputs")
            if isinstance(outputs, dict):
                new_outputs: Dict[str, Any] = {}
                local_renamed: Dict[str, str] = {}
                for var_name, what in outputs.items():
                    canonical_name = None

                    # Attempt to derive canonical name if 'what' looks like field(entity)
                    if isinstance(what, str):
                        m = re.search(r'([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([A-Za-z0-9_]+)\s*\)', what.strip())
                        if m:
                            fld = m.group(1).strip()
                            ent = m.group(2).strip()
                            expr_key = f"{fld.lower()}({ent.lower()})"
                            canonical_from_dsl = expr_to_canonical.get(expr_key)
                            if canonical_from_dsl:
                                canonical_name = canonical_from_dsl

                    # If we found a canonical name, check if current name matches (token conservation)
                    final_name = var_name
                    if canonical_name:
                        if self._names_violate_conservation(var_name, canonical_name):
                            final_name = canonical_name
                            local_renamed[var_name] = final_name

                    new_outputs[final_name] = outputs[var_name]

                if local_renamed:
                    # Apply rename in this step
                    step["outputs"] = new_outputs
                    renamed.update(local_renamed)

        if renamed:
            # Pass 2: rewrite references in all step texts to new names
            for step in steps:
                text = step.get("text")
                if isinstance(text, str) and text:
                    step["text"] = self._rewrite_var_refs(text, renamed)

        return plan, renamed

    @staticmethod
    def _dsl_pairs_from_segments(segments: List[str]) -> List[Tuple[str, str]]:
        """
        Parse field(entity) pairs out of DSL lines.
        """
        pairs: List[Tuple[str, str]] = []
        for seg in segments or []:
            # Find all field(entity) occurrences
            for m in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([A-Za-z0-9_]+)\s*\)', seg):
                field = m.group(1)
                entity = m.group(2)
                pairs.append((field, entity))
        return pairs

    @staticmethod
    def _to_snake(x: str) -> str:
        # Already underscore-delimited entities should pass through; just normalize case and collapse
        s = re.sub(r'[^A-Za-z0-9_]+', '_', x.strip())
        s = re.sub(r'__+', '_', s)
        s = s.strip('_').lower()
        return s

    def _names_violate_conservation(self, current: str, canonical: str) -> bool:
        """
        True if `current` drops tokens present in `canonical` or differs entirely.
        Enforces token conservation of the entity label + field.
        """
        if current == canonical:
            return False

        def toks(s: str) -> List[str]:
            return [t for t in self._to_snake(s).split('_') if t]

        cur_tokens = set(toks(current))
        can_tokens = toks(canonical)

        # All canonical tokens must appear in current; order not enforced here, only coverage.
        return not set(can_tokens).issubset(cur_tokens)

    @staticmethod
    def _rewrite_var_refs(text: str, rename_map: Dict[str, str]) -> str:
        """
        Replace whole-word occurrences of old variable names with new ones.
        """
        out = text
        for old, new in sorted(rename_map.items(), key=lambda kv: -len(kv[0])):  # longest first
            pattern = r'\b' + re.escape(old) + r'\b'
            out = re.sub(pattern, new, out)
        return out

    # ---------- utils ----------

    @staticmethod
    def _sanitize_text(x: Any) -> str:
        s = " ".join(str(x or "").split())
        if s and s[-1] not in ".?!":
            s += "."
        return s
