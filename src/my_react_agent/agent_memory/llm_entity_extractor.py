from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_adapters.llm_base import LLMBase
from agent_memory.entity_extractor import EntityExtractor

from agent_memory.data_structures import (
    Entity,
    Evidence,
    EvidenceExtract,
    GenderEnum,
    new_entity,
    entity_with_evidence,
)

logger = logging.getLogger(__name__)


class LLMEntityExtractor(EntityExtractor):
    _STRIP = "\"' \t\r\n`"
    _MAX_JSON_CHARS = 50_000

    _CONNECTORS = {"in", "of", "the", "and", "&"}

    _QUESTION_STOP = {
        "How", "What", "When", "Where", "Who", "Why",
        "Is", "Are", "Was", "Were", "Do", "Does", "Did",
        "Tell", "Give", "Find", "Show", "Explain",
        "A", "An", "The", "This", "That", "These", "Those",
        "I", "We", "You", "He", "She", "They", "It",
    }

    def __init__(self, llm: LLMBase, *, max_chars: int = 2500, max_entities: int = 5):
        self.llm = llm
        self.max_chars = int(max_chars)
        self.max_entities = int(max_entities)

    def extract_entities(self, text: str) -> List[str]:
        src_full = text or ""
        src = src_full.strip()
        if not src:
            return []

        if len(src) > self.max_chars:
            src = src[: self.max_chars]

        names = self._llm_extract_names(src)

        if not names:
            names = self._regex_fallback_names(src)

        names = self._postprocess_and_rank(names, src)
        return names[: self.max_entities]

    def find_entities(self, text: str) -> List[str]:
        return self.extract_entities(text)

    def run(self, text: str) -> List[str]:
        return self.extract_entities(text)


    def extract(self, text: str) -> Optional[Entity]:
        names = self.extract_entities(text)
        if not names:
            return None

        primary = names[0]
        ent = new_entity(primary)

        ev = Evidence(
            tool="llm_entity_extractor",
            content=(text or ""),
            url=None,
            extracted=EvidenceExtract(fields={"primary": primary}, payload={"entities": names}),
            as_of=datetime.utcnow(),
            confidence=0.6,
        )
        ent = entity_with_evidence(ent, ev)
        return ent

    def _llm_extract_names(self, src: str) -> List[str]:
        schema = (
            '{'
            '"entities":['
            '{"name":"<EXACT substring from INPUT>",'
            '"type":"person|org|location|country|city|work|event|object|concept",'
            '"aliases":["<optional exact substrings>"],'
            '"gender":"male|female|neutral|unknown"}'
            "]}"
        )

        prompt = (
            "You are a deterministic entity extraction function.\n"
            "Extract the MOST IMPORTANT named entities from INPUT.\n\n"
            "RULES (CRITICAL):\n"
            "1) Return STRICT JSON only. No markdown, no commentary.\n"
            "2) Each entity 'name' MUST be an EXACT substring of INPUT.\n"
            "3) Prefer the MOST SPECIFIC mention (e.g. 'Burgau in Styria, Austria' over 'Burgau').\n"
            "4) Only include entities that help answer the question.\n"
            "5) Return at most 5 entities.\n"
            "6) Do not return generic nouns like 'population' unless it is a proper name.\n\n"
            f"JSON SHAPE:\n{schema}\n\n"
            f"INPUT:\n{src}\n"
            "JSON:"
        )

        raw = (self.llm.generate(prompt) or "").strip()
        if raw and len(raw) > self._MAX_JSON_CHARS:
            raw = raw[: self._MAX_JSON_CHARS]

        data = self._safe_parse_json_object(raw)
        if not isinstance(data, dict):
            return []

        ents = data.get("entities", [])
        if not isinstance(ents, list):
            return []

        out: List[str] = []
        for item in ents:
            if isinstance(item, str):
                nm = item.strip(self._STRIP).strip()
                if nm:
                    out.append(nm)
                continue

            if isinstance(item, dict):
                nm = str(item.get("name") or item.get("text") or item.get("canonical_name") or "").strip(self._STRIP).strip()
                if nm:
                    out.append(nm)

                als = item.get("aliases", [])
                if isinstance(als, list):
                    for a in als:
                        if isinstance(a, str):
                            aa = a.strip(self._STRIP).strip()
                            if aa:
                                out.append(aa)

        return out


    def _safe_parse_json_object(self, raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None

        s0 = raw.strip()

        s = re.sub(r"(?is)```(?:json|javascript|js|python|txt)?\s*|\s*```", "", s0).strip()

        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s, flags=re.IGNORECASE)

        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        try:
            i, j = s.find("{"), s.rfind("}")
            if i != -1 and j != -1 and j > i:
                obj = json.loads(s[i : j + 1])
                if isinstance(obj, dict):
                    return obj
        except Exception:
            pass

        try:
            start = s.find("{")
            if start != -1:
                depth = 0
                in_str = False
                esc = False
                for k in range(start, len(s)):
                    ch = s[k]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                    else:
                        if ch == '"':
                            in_str = True
                        elif ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                candidate = s[start : k + 1]
                                obj = json.loads(candidate)
                                if isinstance(obj, dict):
                                    return obj
                                break
        except Exception:
            pass

        return None


    def _regex_fallback_names(self, src: str) -> List[str]:

        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+|[,.-]", src)
        if not tokens:
            return []

        phrases: List[str] = []
        i = 0
        while i < len(tokens):
            t = tokens[i]

            if not self._looks_like_entity_token(t):
                i += 1
                continue

            buf = [t]
            j = i + 1
            while j < len(tokens):
                nxt = tokens[j]

                if nxt in {",", "-"}:
                    buf.append(nxt)
                    j += 1
                    continue

                if self._looks_like_entity_token(nxt):
                    buf.append(nxt)
                    j += 1
                    continue

                if nxt.lower() in self._CONNECTORS:
                    buf.append(nxt)
                    j += 1
                    continue

                break

            phrase = self._join_phrase(buf).strip()
            phrase = phrase.strip(self._STRIP).strip()

            if phrase and phrase.split()[0] not in self._QUESTION_STOP:
                phrases.append(phrase)

            i = j

        return phrases

    def _looks_like_entity_token(self, tok: str) -> bool:
        if not tok or tok in {",", ".", "-"}:
            return False
        return (tok[0].isupper() and any(ch.isalpha() for ch in tok)) or tok.isupper()

    def _join_phrase(self, toks: List[str]) -> str:
        out: List[str] = []
        for t in toks:
            if t in {",", ".", "-"}:
                if out:
                    out[-1] = out[-1] + t
                else:
                    out.append(t)
            else:
                out.append(t)
        return " ".join(out)


    def _postprocess_and_rank(self, names: List[str], src: str) -> List[str]:
        if not names:
            return []

        src_norm = src

        cleaned: List[str] = []
        seen: Set[str] = set()

        for n in names:
            if not isinstance(n, str):
                continue
            x = n.strip(self._STRIP).strip()
            if not x:
                continue

            if x not in src_norm:
                idx = self._find_case_insensitive(src_norm, x)
                if idx == -1:
                    continue
                x = src_norm[idx : idx + len(x)]

            key = x.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)

            first = x.split()[0] if x.split() else ""
            if first in self._QUESTION_STOP:
                continue

            cleaned.append(x)

        cleaned.sort(key=lambda s: (len(s.split()), len(s)), reverse=True)
        return cleaned

    def _find_case_insensitive(self, haystack: str, needle: str) -> int:
        if not haystack or not needle:
            return -1
        try:
            m = re.search(re.escape(needle), haystack, flags=re.IGNORECASE)
            return m.start() if m else -1
        except Exception:
            return -1
