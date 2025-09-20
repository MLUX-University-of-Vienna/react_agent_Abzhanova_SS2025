# agent_memory/pronouns_resolution/llm_entity_extractor.py
from __future__ import annotations

import json, re
from typing import Optional

from .entity_extractor import EntityExtractor
from agent_memory.entity import Mention
from agent_memory.gender_enum import GenderEnum
from llm_adapters.llm_base import LLMBase


class LLMEntityExtractor(EntityExtractor):
    def __init__(self, llm: LLMBase, max_chars: int = 3000):
        self.llm = llm
        self.max_chars = max_chars

    def extract(self, text: str, query_id: Optional[str] = None, subquery_id: Optional[str] = None) -> Optional[Mention]:
        src = text.strip()
        if not src:
            return None
        if len(src) > self.max_chars:
            src = src[: self.max_chars]

        prompt = (
            "Extract the SINGLE most relevant referent from the sentence.\n"
            "Return STRICT JSON ONLY:\n"
            '{ "span": "<exact substring from input>", "label": "<coarse label>", '
            '"start": <int>, "end": <int>, "gender": "<male|female|neutral|unknown|null>" }\n'
            'If none, return {"span":"NONE","label":"none","start":-1,"end":-1,"gender":"unknown"}\n\n'
            f"INPUT: {src!r}\nJSON:"
        )
        raw = (self.llm.generate(prompt) or "").strip()
        ent = self._parse_span(raw, src)
        if ent:
            return ent
        return self._fallback_heuristics(src)

    def _parse_span(self, raw: str, src: str) -> Optional[Mention]:
        try:
            data = json.loads(raw)
        except Exception:
            s, e = raw.find("{"), raw.rfind("}")
            if s == -1 or e == -1 or e <= s:
                return None
            try:
                data = json.loads(raw[s : e + 1])
            except Exception:
                return None

        span = str(data.get("span") or "").strip()
        if not span or span.upper() == "NONE":
            return None

        try:
            start = int(data.get("start", -1))
            end = int(data.get("end", -1))
        except Exception:
            start, end = -1, -1

        if start < 0 or end < 0 or src[start:end] != span:
            cand = span.strip("\"' ").strip()
            idx = src.find(cand) if cand else -1
            if idx == -1:
                return None
            start, end, span = idx, idx + len(cand), cand

        label_raw = (data.get("label") or None)
        graw = (data.get("gender") or "unknown").strip().lower()
        gender = {
            "male": GenderEnum.MALE,
            "female": GenderEnum.FEMALE,
            "neutral": GenderEnum.NEUTRAL,
            "unknown": GenderEnum.UNKNOWN,
            "null": GenderEnum.UNKNOWN,
            "none": GenderEnum.UNKNOWN,
        }.get(graw, GenderEnum.UNKNOWN)

        if gender == GenderEnum.UNKNOWN and (str(label_raw or "").strip().lower() in {"person", "human"}):
            window = src[max(0, start - 80) : end + 80].lower()
            if re.search(r"\bmr\.?\b", window):
                gender = GenderEnum.MALE
            elif re.search(r"\b(mrs|ms|miss)\.?\b", window):
                gender = GenderEnum.FEMALE
            else:
                male_pron = re.search(r"\b(he|him|his)\b", window)
                female_pron = re.search(r"\b(she|her|hers)\b", window)
                if male_pron and not female_pron:
                    gender = GenderEnum.MALE
                elif female_pron and not male_pron:
                    gender = GenderEnum.FEMALE

        return Mention(text=span, start=start, end=end, gender=gender, entity_id=None)

    def _fallback_heuristics(self, src: str) -> Optional[Mention]:
        s = src

        def pack(span: str) -> Mention:
            i = s.find(span)
            return Mention(text=span, start=i, end=i + len(span), gender=GenderEnum.UNKNOWN, entity_id=None)

        m = re.search(r"\b(what|who|where)\s+(?:is|are)\s+(.+?)(?=\s+(?:and|or)\b|[?.!,]|$)", s, flags=re.I)
        if m:
            return pack(m.group(2).strip())

        m = re.search(r"\b(synonym|definition|meaning)\s+of\s+(.+?)(?=\s+(?:and|or)\b|[?.!,]|$)", s, flags=re.I)
        if m:
            return pack(m.group(2).strip())

        m = re.search(r"\b(population|price|capital)\s+of\s+(.+?)(?=\s+(?:and|or)\b|[?.!,]|$)", s, flags=re.I)
        if m:
            return pack(m.group(2).strip())

        q = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", s)
        if q:
            q.sort(key=lambda x: len(x), reverse=True)
            return pack(q[0])

        m = re.findall(r'"([^"]+)"|\'([^\']+)\'', s)
        if m:
            span = next((x for pair in m for x in pair if x), "").strip()
            if span:
                return pack(span)

        return None
