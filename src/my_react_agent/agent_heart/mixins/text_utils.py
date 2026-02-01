from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

from ...agent_core.agent_actions.action import Action

from ..logger import logger


class TextUtilsMixin:
    def _sanitise_to_text(self, s: Any) -> str:
        try:
            if isinstance(s, (dict, list, tuple)):
                s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
        s2 = (str(s) if s is not None else "").replace("\n", " ").replace("\r", " ").strip()
        return re.sub(r"\s{2,}", " ", s2)

    def _ensure_sentence(self, s: str) -> str:
        x = self._sanitise_to_text(s)
        if not x:
            return ""
        if x[-1] not in ".!?":
            x = x + "."
        return x

    def _balance_json_closers(self, s: str) -> str:
        stack: List[str] = []
        in_str = False
        esc = False

        for ch in s or "":
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    top = stack[-1]
                    if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                        stack.pop()

        if stack:
            closers = "".join("}" if op == "{" else "]" for op in reversed(stack))
            return (s or "") + closers
        return s or ""

    def _safe_parse_json_object(self, raw: str, *, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        fb = fallback if isinstance(fallback, dict) else {}

        s0 = (raw or "").strip()
        s = re.sub(r"(?is)```(?:json|javascript|js|python|txt)?\s*|\s*```", "", s0).strip()

        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s, flags=re.IGNORECASE)

        def try_load(x: str) -> Optional[Dict[str, Any]]:
            try:
                obj = json.loads(x)
                return obj if isinstance(obj, dict) else None
            except JSONDecodeError as e:
                tail = x[max(0, len(x) - 200) :]
                logger.warning(
                    "[_safe_parse_json_object] JSONDecodeError=%s pos=%d tail=%r",
                    e.msg,
                    e.pos,
                    tail,
                )
                return None
            except Exception as e:
                logger.warning("[_safe_parse_json_object] json.loads error=%r", e)
                return None

        obj = try_load(s)
        if obj is not None:
            return obj

        if "{" in s:
            s1 = s[s.find("{") : (s.rfind("}") + 1 if "}" in s else len(s))]
            obj = try_load(s1)
            if obj is not None:
                return obj
            s1b = self._balance_json_closers(s1)
            obj = try_load(s1b)
            if obj is not None:
                return obj

        sb = self._balance_json_closers(s)
        obj = try_load(sb)
        if obj is not None:
            return obj

        logger.warning("[_safe_parse_json_object] json parse failed; returning fallback keys=%s", list(fb.keys()))
        return fb

    def _actions_catalog_text(self, actions: List[Action]) -> str:
        if not actions:
            return "(none)"
        entries: List[str] = []
        for action in actions:
            try:
                entries.append(action.to_catalog_entry())
            except Exception as e:
                entries.append(f"- {getattr(action, 'name', 'UNKNOWN')} (error: {e})")
        return "\n".join(entries)
