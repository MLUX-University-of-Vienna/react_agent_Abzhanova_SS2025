from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_CODE_FENCE_MARKERS_RE = re.compile(r"(?is)```(?:json|javascript|js|python|txt)?\s*|```")

_PREFIX_NOISE_RE = re.compile(
    r"(?is)^\s*(here\s+is\s+the\s+(json|recommended\s+next\s+action).*?:\s*|json\s*:\s*)"
)

_EXPLANATION_BLOCK_RE = re.compile(r"(?is)\}\s*(?:explanation|reasoning)\s*:.*$")


def _safe_text(obj: Any) -> str:
    if obj is None:
        return ""
    try:
        s = str(obj)
    except Exception:
        return ""
    s = s.replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def _round2(x: float) -> float:
    return float(f"{_clamp01(x):.2f}")


def _strip_code_fences_keep_content(s: str) -> str:
    return _CODE_FENCE_MARKERS_RE.sub("", s or "").strip()


def _normalise_jsonish(s: str) -> str:
    if not s:
        return ""

    s = _PREFIX_NOISE_RE.sub("", s).strip()

    try:
        s = _EXPLANATION_BLOCK_RE.sub("}", s).strip()
    except Exception:
        pass

    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s, flags=re.IGNORECASE)

    s = re.sub(r",\s*([}\]])", r"\1", s)

    return s.strip()


def _find_first_balanced_json_object(s: str) -> Optional[str]:
    if not s:
        return None
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
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
                    return s[start : i + 1]
    return None


def _try_repair_truncated_json_object(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if not s.startswith("{"):
        return None

    depth = 0
    in_str = False
    esc = False
    for ch in s:
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
                if depth < 0:
                    depth = 0
    if depth <= 0:
        return None
    return s + ("}" * depth)


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    s = (raw or "").strip()
    if not s:
        return None

    s = _strip_code_fences_keep_content(s)
    s = _normalise_jsonish(s)

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    candidate = _find_first_balanced_json_object(s)
    if candidate:
        candidate = _normalise_jsonish(candidate)
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    m = re.search(r"(?is)(\{[\s\S]*\})", s)
    if m:
        candidate = _normalise_jsonish(m.group(1))
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    repaired = _try_repair_truncated_json_object(s)
    if repaired:
        repaired = _normalise_jsonish(repaired)
        try:
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    return None


def _serialise_tool_result(result: Any) -> str:
    if result is None:
        return ""

    obs_list = getattr(result, "observations", None)
    if isinstance(obs_list, list) and obs_list:
        parts: List[str] = []
        for i, obs in enumerate(obs_list[:6], start=1):
            rt = getattr(obs, "raw_text", None)
            title = getattr(obs, "title", None)
            src = getattr(obs, "source", None)
            url = getattr(obs, "source_url", None)

            line: List[str] = [f"[OBS {i}]"]
            if isinstance(title, str) and title.strip():
                line.append(f"title={title.strip()!r}")
            if isinstance(src, str) and src.strip():
                line.append(f"source={src.strip()!r}")
            if isinstance(url, str) and url.strip():
                line.append(f"url={url.strip()!r}")

            text = (rt or "")
            if isinstance(text, str):
                text = text.strip()
                if len(text) > 2000:
                    text = text[:2000].rstrip() + "..."
                if text:
                    line.append(f"text={text!r}")

            parts.append(" ".join(line).strip())

        md = getattr(result, "metadata", None)
        if isinstance(md, dict) and md:
            keep = {k: md.get(k) for k in ("tool", "primary_url", "page_title", "subject") if k in md}
            if keep:
                parts.append(f"[METADATA] {keep!r}")

        return "\n".join(parts).strip()

    if hasattr(result, "text") and isinstance(getattr(result, "text"), str):
        return str(getattr(result, "text"))
    if hasattr(result, "content") and isinstance(getattr(result, "content"), str):
        return str(getattr(result, "content"))

    return _safe_text(result)


def _coerce_score_0_1(x: Any, fallback: float) -> float:
    try:
        v = float(x)
    except Exception:
        return _clamp01(fallback)

    if 1.0 < v <= 10.0:
        v = v / 10.0
    return _clamp01(v)


def _pick_reason(obj: Dict[str, Any], default: str) -> str:
    if not isinstance(obj, dict):
        return default
    for k in ("reason", "evaluation", "explanation", "comment"):
        val = obj.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return default