from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent_memory.data_structures import (
    Step,
    Query,
    Entity,
    query_replace_step,
    query_upsert_entity,
    entity_with_evidence,
    new_entity,
)

from agent_heart.logger import logger

@dataclass
class StepTraceRecorder:
    agent: Any

    def update_previous_step_results(self, step: Step) -> None:
        try:
            obs = (step.result.observation if step.result else "") or ""
            ev_snip = ""
            if step.evidence:
                try:
                    ev_snip = step.evidence[-1].content
                except Exception:
                    ev_snip = ""

            extracted_facts: List[Dict[str, Any]] = []
            try:
                ef = getattr(step.result, "extracted_facts", None) if step.result else None
                if isinstance(ef, (list, tuple)):
                    extracted_facts = [x for x in ef if isinstance(x, dict)]
            except Exception:
                extracted_facts = []

            record = {
                "index": step.plan.index,
                "instruction": step.plan.instruction,
                "observation": obs,
                "evidence": ev_snip,
                "raw_result": obs,
                "extracted_facts": extracted_facts,
            }

            self.agent._previous_step_results = [
                r for r in self.agent._previous_step_results
                if not (isinstance(r, dict) and r.get("index") == step.plan.index)
            ]
            self.agent._previous_step_results.append(record)

            logger.info("[ReActAgent._update_previous_step_results] Stored result for step %d", step.plan.index)
        except Exception as e:
            logger.warning("[ReActAgent._update_previous_step_results] failed error=%r", e)

    def extract_step_evidence_snippet(self, step: Step) -> str:
        if not step.evidence:
            return ""
        try:
            return self.agent._sanitise_to_text(step.evidence[-1].content)
        except Exception:
            return ""

    def _safe_extract_entity_names(self, text: str) -> List[str]:
        """
        Best-effort, non-invasive entity extraction.
        This does NOT change prompts/logs and only runs if extractor exposes a known method.
        """
        t = (text or "").strip()
        if not t:
            return []

        extractor = getattr(self.agent, "entity_extractor", None)
        if extractor is None:
            return []

        for method_name in ("extract", "extract_entities", "find_entities", "run"):
            fn = getattr(extractor, method_name, None)
            if callable(fn):
                try:
                    out = fn(t)
                    if isinstance(out, list):
                        names: List[str] = []
                        for item in out:
                            if isinstance(item, str):
                                if item.strip():
                                    names.append(item.strip())
                            elif isinstance(item, dict):
                                nm = item.get("canonical_name") or item.get("name") or item.get("text")
                                if isinstance(nm, str) and nm.strip():
                                    names.append(nm.strip())
                        return list(dict.fromkeys(names))
                    if isinstance(out, dict):
                        ents = out.get("entities")
                        if isinstance(ents, list):
                            names = []
                            for item in ents:
                                if isinstance(item, str) and item.strip():
                                    names.append(item.strip())
                                elif isinstance(item, dict):
                                    nm = item.get("canonical_name") or item.get("name") or item.get("text")
                                    if isinstance(nm, str) and nm.strip():
                                        names.append(nm.strip())
                            return list(dict.fromkeys(names))
                except Exception:
                    return []
        return []

    def _upsert_entity_from_step(self, q: Query, name: str, step: Step) -> Query:
        canonical = (name or "").strip()
        if not canonical:
            return q

        existing: Optional[Entity] = None
        for e in (q.entities or ()):
            if (e.canonical_name or "").strip().lower() == canonical.lower():
                existing = e
                break

        ent = existing or new_entity(canonical)

        if step.evidence:
            try:
                ev = step.evidence[-1]
                ent = entity_with_evidence(ent, ev)
            except Exception:
                pass

        return query_upsert_entity(q, ent)

    def persist_step_to_query_memory(self, *, step: Step) -> None:
        try:
            if not hasattr(self.agent, "query_memory") or self.agent.query_memory is None:
                return
            q = self.agent.query_memory.current_query_model
            if not q:
                return

            before = []
            try:
                before = list(q.entities or ())
            except Exception:
                before = []

            before_keys = {
                (getattr(e, "canonical_name", "") or "").strip().lower()
                for e in before
                if (getattr(e, "canonical_name", "") or "").strip()
            }

            q2 = query_replace_step(q, step)

            obs_text = ""
            try:
                obs_text = (step.result.observation if step.result else "") or ""
            except Exception:
                obs_text = ""

            candidates: List[str] = []
            candidates.extend(self._safe_extract_entity_names(step.plan.instruction))
            candidates.extend(self._safe_extract_entity_names(obs_text))

            logger.info(
                "[EntityExtract] step=%d raw_candidates=%s",
                step.plan.index,
                candidates,
            )

            seen = set()
            uniq = []
            for n in candidates:
                k = (n or "").strip().lower()
                if not k or k in seen:
                    continue
                seen.add(k)
                uniq.append(n.strip())
            
            logger.info(
                "[EntityExtract] step=%d uniq=%s",
                step.plan.index,
                uniq,
            )

            current_keys = set(before_keys)
            new_names: List[str] = []

            existing_names: List[str] = []
            for name in uniq[:12]:
                key = (name or "").strip().lower()
                if not key:
                    continue

                if key in current_keys:
                    existing_names.append(name)
                else:
                    new_names.append(name)

                q2 = self._upsert_entity_from_step(q2, name, step)
                current_keys.add(key)

            after = []
            try:
                after = list(q2.entities or ())
            except Exception:
                after = []

            logger.info(
                "[EntityExtract] step=%d added=%s existing=%s total_entities=%d",
                step.plan.index,
                new_names,
                existing_names,
                len(after),
            )

            self.agent.query_memory.current_query_model = q2

        except Exception as e:
            logger.exception("[EntityExtract] persist_step_to_query_memory failed: %r", e)