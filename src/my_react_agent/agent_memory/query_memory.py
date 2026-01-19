from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from .data_structures import (
    Query,
    QueryStatus,
    Step,
    StepPlan,
    StepStatus,
    Entity,
    query_upsert_entity,
    new_query,
    new_step,
)

logger = logging.getLogger(__name__)


class QueryMemory:
    def __init__(self):
        self.original_query_text: str = ""

        self.current_query: Optional[Query] = None

        self.global_plan: List[Dict[str, Any]] = []
        self.step_trace: List[Dict[str, Any]] = []
        self.transcript_lines: List[str] = []

        self._step_extracted_entities: Dict[int, List[Dict[str, Any]]] = {}
        self._legacy_flat_entity_store_view: Dict[str, Any] = {}

        logger.info("QueryMemory initialised (no active query)")


    @property
    def current_query_model(self) -> Optional[Query]:
        return self.current_query

    @current_query_model.setter
    def current_query_model(self, value: Optional[Query]) -> None:
        self.current_query = value

    def start_new(self, query_text: str) -> None:
        self.original_query_text = query_text or ""
        self.current_query = new_query(self.original_query_text)
        self.current_query = replace(self.current_query, status=QueryStatus.PLANNED)

        self.clear_trace()

        q = self.current_query
        assert q is not None
        logger.info(
            "New query session started id=%s text_length=%d",
            str(q.id),
            len(self.original_query_text),
        )

    def save_global_plan(self, planned_steps: Any) -> None:
        self._require_active_query()
        q = self.current_query
        assert q is not None

        plan_payload = self._normalise_plan_payload(planned_steps)
        self.global_plan = plan_payload if isinstance(plan_payload, list) else [plan_payload]

        placeholder_steps: List[Step] = []
        for item in (self.global_plan or []):
            if not isinstance(item, dict):
                continue

            step_index = self._safe_int(item.get("index", 0), default=0)
            if step_index <= 0:
                continue

            instruction = str(item.get("instruction", "") or "")
            needed_facts = list(item.get("needed_facts", []) or [])
            depends_on = list(item.get("depends_on", []) or [])

            plan = StepPlan(
                index=step_index,
                instruction=instruction,
                needed_facts=[str(x) for x in needed_facts if str(x).strip()],
                depends_on=[
                    int(x)
                    for x in depends_on
                    if isinstance(x, (int, str)) and str(x).strip().isdigit()
                ],
            )

            s = new_step(plan)
            s = replace(s, status=StepStatus.PLANNED, started_at=None, finished_at=None)
            placeholder_steps.append(s)

        if placeholder_steps:
            placeholder_steps.sort(key=lambda s: s.plan.index)
            self.current_query = replace(q, steps=tuple(placeholder_steps))

        logger.debug("Saved global plan steps=%d", len(self.global_plan))

    def save_step_trace(
        self,
        *,
        index: int,
        instruction: str = "",
        needed_entities: Optional[List[str]] = None,
        depends_on: Optional[List[int]] = None,
        thought: str = "",
        action: str = "",
        tool_name: str = "",
        tool_input: str = "",
        observation: str = "",
        evidence_snippet: str = "",
        raw_result: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._require_active_query()
        q = self.current_query
        assert q is not None

        step_index = int(index)

        trace_record = {
            "index": step_index,
            "instruction": instruction or "",
            "needed_entities": list(needed_entities or []),
            "depends_on": list(depends_on or []),
            "thought": thought or "",
            "action": action or "",
            "tool_name": tool_name or "",
            "tool_input": tool_input or "",
            "observation": observation or "",
            "evidence_snippet": evidence_snippet or "",
            "raw_result": raw_result or "",
            "extra": extra or {},
        }

        replaced_existing = False
        for i, record in enumerate(self.step_trace):
            if isinstance(record, dict) and int(record.get("index", -1)) == step_index:
                self.step_trace[i] = trace_record
                replaced_existing = True
                break
        if not replaced_existing:
            self.step_trace.append(trace_record)

        self.step_trace.sort(key=lambda r: int(r.get("index", 0)))

        steps_list = list(q.steps or ())
        step_obj = self._find_step_by_index(steps_list, step_index)

        if step_obj is None:
            plan = StepPlan(
                index=step_index,
                instruction=instruction or "",
                needed_facts=list(needed_entities or []),
                depends_on=list(depends_on or []),
            )
            step_obj = new_step(plan)
            step_obj = replace(step_obj, status=StepStatus.RUNNING, started_at=None, finished_at=None)

        plan2 = step_obj.plan
        if instruction:
            plan2 = replace(plan2, instruction=instruction)
        if needed_entities is not None:
            plan2 = replace(plan2, needed_facts=list(needed_entities))
        if depends_on is not None:
            plan2 = replace(plan2, depends_on=list(depends_on))

        step_obj = replace(step_obj, plan=plan2)
        steps_list = self._upsert_step_by_id_or_index(steps_list, step_obj)
        steps_list.sort(key=lambda s: s.plan.index)

        self.current_query = replace(q, steps=tuple(steps_list))

        logger.debug("Saved step trace index=%d (replaced=%s)", step_index, replaced_existing)

    def set_transcript(self, transcript_lines: List[str]) -> None:
        self._require_active_query()
        self.transcript_lines = [str(x or "") for x in (transcript_lines or [])]
        logger.debug("Set transcript lines total=%d", len(self.transcript_lines))

    def append_transcript_line(self, line: str) -> None:
        self._require_active_query()
        self.transcript_lines.append(str(line or ""))
        logger.debug("Appended transcript line (total=%d)", len(self.transcript_lines))

    def clear_trace(self) -> None:
        self.global_plan = []
        self.step_trace = []
        self.transcript_lines = []
        self._step_extracted_entities = {}
        self._legacy_flat_entity_store_view = {}
        logger.debug("Cleared global_plan/step_trace/transcript/compat caches")

    def save_step_extracted_entities(self, extracted_by_step: Dict[int, Any]) -> None:
        self._require_active_query()

        normalised: Dict[int, List[Dict[str, Any]]] = {}
        if isinstance(extracted_by_step, dict):
            for k, v in extracted_by_step.items():
                step_index = self._safe_int(k, default=None)
                if step_index is None:
                    continue
                if isinstance(v, list):
                    normalised[step_index] = [x for x in v if isinstance(x, dict)]

        self._step_extracted_entities = normalised
        logger.debug("Saved step_extracted_entities steps=%d", len(self._step_extracted_entities))

    def save_step_extracted_facts(self, step_extracted_facts: Dict[int, Any]) -> None:
        self.save_step_extracted_entities(step_extracted_facts)

    def save_legacy_flat_entity_store(self, flat_store: Dict[str, Any]) -> None:
        self._require_active_query()
        self._legacy_flat_entity_store_view = dict(flat_store) if isinstance(flat_store, dict) else {}
        logger.debug("Saved legacy_flat_entity_store keys=%d", len(self._legacy_flat_entity_store_view))

    def save_fact_store(self, fact_store: Dict[str, Any]) -> None:
        self.save_legacy_flat_entity_store(fact_store)

    def upsert_entity(self, entity: Entity) -> None:
        self._require_active_query()
        q = self.current_query
        assert q is not None
        self.current_query = query_upsert_entity(q, entity)

    def search_entity_memory(self, text: str) -> Optional[Dict[str, Any]]:
        q = self.current_query_model
        if not q or not q.entities:
            return None

        query_text_lower = (text or "").strip().lower()
        if not query_text_lower:
            return None

        best_payload: Optional[Dict[str, Any]] = None
        best_score: float = -1.0

        for entity in q.entities or ():
            name = (entity.canonical_name or "").strip()
            if not name:
                continue

            name_lower = name.lower()
            alias_lowers = {a.strip().lower() for a in (entity.aliases or set()) if isinstance(a, str)}

            entity_hit = (name_lower in query_text_lower) or any(al in query_text_lower for al in alias_lowers)

            if not entity_hit:
                continue

            ev = entity.evidence[-1] if entity.evidence else None
            obs = ""
            conf = 0.6
            extracted_facts: List[Dict[str, Any]] = []

            if ev is not None:
                snippet = (ev.content or "").strip()
                obs = snippet if snippet else f"Found entity: {name}"
                conf = float(min(0.95, max(0.1, ev.confidence)))

                try:
                    payload = getattr(ev.extracted, "payload", None) if ev.extracted is not None else None
                    if isinstance(payload, dict):
                        fx = payload.get("facts", []) or []
                        if isinstance(fx, list):
                            extracted_facts = [x for x in fx if isinstance(x, dict)]
                except Exception:
                    extracted_facts = []
            else:
                obs = f"Found entity: {name}"
                conf = 0.4

            score = 1.0 + (0.5 if ev is not None else 0.0) + conf

            payload_out = {
                "observation": obs,
                "confidence": conf,
                "extracted": {
                    "entity_id": str(entity.entity_id),
                    "entity": name,
                    "evidence_tool": (ev.tool if ev is not None else None),
                    "evidence_url": (ev.url if ev is not None else None),
                    "as_of": (ev.as_of.isoformat() if ev is not None else None),
                },
                "extracted_facts": extracted_facts or [
                    {"key": "evidence", "value": obs, "number": None, "unit": "", "provenance": {"entity": name}}
                ],
            }

            if score > best_score:
                best_score = score
                best_payload = payload_out

        return best_payload


    def get_context_for_step_task(self, step_task: str) -> str:
        if not self.current_query:
            logger.info("get_context_for_step_task called with no current_query; returning empty")
            return ""

        lines: List[str] = []
        for step in self.current_query.steps or ():
            if step.plan.instruction == step_task:
                continue

            obs = ""
            if step.result is not None:
                obs = str(step.result.observation or "")
            lines.append(f"Q: {step.plan.instruction}\nA: {obs}")
        return "\n".join(lines[-3:])

    def _require_active_query(self) -> None:
        if not self.current_query:
            raise RuntimeError("No active current_query. Call start_new() first.")

    def _normalise_plan_payload(self, planned_steps: Any) -> Any:
        if planned_steps is None:
            return []
        if isinstance(planned_steps, list):
            out: List[Any] = []
            for item in planned_steps:
                out.append(item if isinstance(item, dict) else self._plan_step_to_dict(item))
            return out
        if isinstance(planned_steps, dict):
            return planned_steps
        return self._plan_step_to_dict(planned_steps)

    def _plan_step_to_dict(self, step_obj: Any) -> Dict[str, Any]:
        if step_obj is None:
            return {}
        if isinstance(step_obj, dict):
            return step_obj

        get_attr = lambda k, default=None: getattr(step_obj, k, default)
        payload = {
            "index": get_attr("index", None),
            "instruction": get_attr("instruction", "") or "",
            "needed_facts": list(get_attr("needed_facts", []) or []),
            "depends_on": list(get_attr("depends_on", []) or []),
        }
        return {k: v for k, v in payload.items() if v is not None}

    def _build_prior_step_pairs(self) -> List[Tuple[str, str]]:
        if not self.current_query:
            return []
        pairs: List[Tuple[str, str]] = []
        for step in self.current_query.steps or ():
            task = step.plan.instruction
            obs = ""
            if step.result is not None:
                obs = str(step.result.observation or "")
            if task:
                pairs.append((task, obs))
        return pairs

    def _find_step_by_index(self, steps: List[Step], index: int) -> Optional[Step]:
        for step in steps:
            if step.plan.index == index:
                return step
        return None

    def _upsert_step_by_id_or_index(self, steps: List[Step], updated_step: Step) -> List[Step]:
        for i, step in enumerate(steps):
            if step.id == updated_step.id:
                steps[i] = updated_step
                return steps
        for i, step in enumerate(steps):
            if step.plan.index == updated_step.plan.index:
                steps[i] = updated_step
                return steps
        steps.append(updated_step)
        return steps

    def _safe_int(self, value: Any, *, default: Optional[int]) -> Optional[int]:
        try:
            return int(value)
        except Exception:
            return default
