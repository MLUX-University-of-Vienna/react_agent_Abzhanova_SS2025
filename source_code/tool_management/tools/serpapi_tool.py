# tool_management/tools/serpapi_tool.py
import logging
import math
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse

from serpapi.google_search import GoogleSearch

from .agent_tool import AgentTool
from agent_core.models import ToolResponse, Observation

logger = logging.getLogger(__name__)


class SerpAPITool(AgentTool):
    def __init__(self, api_key: str, max_results: int = 5, hl: str = "en", gl: Optional[str] = None):
        self.api_key = api_key
        self.max_results = int(max_results)
        self.hl = hl
        self.gl = gl
        self._stop = {
            "the",
            "a",
            "an",
            "to",
            "of",
            "in",
            "for",
            "and",
            "or",
            "on",
            "with",
            "by",
            "is",
            "are",
            "at",
            "from",
            "as",
            "that",
            "this",
            "these",
            "those",
            "be",
            "was",
            "were",
            "it",
            "its",
            "about",
            "how",
            "what",
            "when",
            "where",
            "who",
            "why",
            "which",
            "can",
            "i",
            "you",
        }
        logger.info(f"[SerpAPITool.__init__] max_results={self.max_results} hl={self.hl} gl={self.gl}")

    @property
    def name(self) -> str:
        return "google_search"

    @property
    def description(self) -> str:
        return "Performs live web or news searches and returns top-ranked results."

    @property
    def refiner_instructions(self) -> str:
        return (
            "Output one line starting with 'web:' or 'news:' followed by the exact search string. "
            "Optionally append 'time:past1h|past24h|past7d|past30d'. "
            "Preserve quoted phrases verbatim. Do not invent filters. Do not compute."
        )

    @property
    def refiner_input_format(self) -> str:
        return "web: <query>  |  news: <query> [ time:past1h|past24h|past7d|past30d ]"

    @property
    def refiner_input_regex(self) -> Optional[str]:
        return r"^(?:web|news):\s\S.*?(?:\s+time:(?:past1h|past24h|past7d|past30d))?$"

    @property
    def refiner_forbidden(self) -> str:
        return "multiple_lines, json, bare_number_only, computed_result"

    @property
    def refiner_examples(self) -> str:
        return "\n".join(
            [
                'web: site:statistik.at "Burgau" Einwohner',
                'news: "ECB rate decision" time:past7d',
                'web: intitle:"OpenGL error" driver fix',
            ]
        )

    def execute(self, tool_input: str) -> ToolResponse:
        try:
            logger.debug(f"[SerpAPITool.execute] tool_input={tool_input!r}")
            mode, query, tbs = self._parse_tool_input(tool_input)
            logger.debug(f"[SerpAPITool.execute] parsed mode={mode} query={query!r} tbs={tbs}")
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "hl": self.hl,
                "num": 20,
            }
            if self.gl:
                params["gl"] = self.gl
            if mode == "news":
                params["tbm"] = "nws"
            if tbs:
                params["tbs"] = tbs
            params_log = {k: v for k, v in params.items() if k != "api_key"}
            logger.info(f"[SerpAPITool.execute] request_params={params_log}")
            data = GoogleSearch(params).get_dict()
            logger.debug(f"[SerpAPITool.execute] serpapi_keys={list(data.keys())}")
            candidates: List[Dict] = []
            if mode == "news":
                candidates = data.get("news_results", []) or []
            else:
                if data.get("answer_box"):
                    candidates.append(self._normalize_special_answer(data["answer_box"], "answer_box"))
                if data.get("knowledge_graph"):
                    candidates.append(self._normalize_special_answer(data["knowledge_graph"], "knowledge_graph"))
                candidates.extend(data.get("organic_results", []) or [])
            if not candidates:
                logger.warning("[SerpAPITool.execute] no_candidates")
                return ToolResponse(
                    success=True,
                    observations=[
                        Observation(raw_text="No results found.", source="serpapi", observed_at=datetime.utcnow())
                    ],
                    metadata={"tool": self.name, "mode": mode},
                )
            ranked = self._rank_candidates(query, candidates)
            deduped = self._dedupe_by_domain_and_title(ranked)
            top = deduped[: self.max_results]
            observations: List[Observation] = []
            for idx, (item, score) in enumerate(top, start=1):
                title = item.get("title") or item.get("source") or "No title"
                snippet = item.get("snippet") or item.get("description") or ""
                link = item.get("link") or item.get("url")
                obs = Observation(
                    raw_text=f"{title}\n{snippet}".strip(),
                    title=title,
                    source="serpapi",
                    source_url=link,
                    observed_at=datetime.utcnow(),
                    meta={
                        "score": round(float(score), 3),
                        "position": item.get("position"),
                        "domain": self._domain_of(link) if link else None,
                        "mode": mode,
                    },
                )
                observations.append(obs)
            logger.info(f"[SerpAPITool.execute] observations_emitted={len(observations)}")
            return ToolResponse(
                success=True,
                observations=observations,
                metadata={"tool": self.name, "mode": mode, "raw_count": len(candidates)},
            )
        except Exception as e:
            logger.exception("[SerpAPITool.execute] search_failed")
            return ToolResponse(
                success=False,
                observations=[
                    Observation(raw_text=f"Search failed: {str(e)}", source="serpapi", observed_at=datetime.utcnow())
                ],
                metadata={"tool": self.name, "error": repr(e)},
            )

    def _parse_tool_input(self, s: str) -> Tuple[str, str, Optional[str]]:
        s = (s or "").strip()
        m = re.match(r"^(web|news):\s*(.*)$", s, flags=re.I)
        if not m:
            return "web", self._strip_outer_quotes(s), None
        mode = m.group(1).lower()
        tail = m.group(2).strip()
        t = re.search(r"\s+time:(past1h|past24h|past7d|past30d)\s*$", tail, flags=re.I)
        tbs = None
        if t:
            tail = tail[: t.start()].rstrip()
            tbs = {"past1h": "qdr:h", "past24h": "qdr:d", "past7d": "qdr:w", "past30d": "qdr:m"}[t.group(1).lower()]
        return mode, self._strip_outer_quotes(tail), tbs

    def _strip_outer_quotes(self, q: str) -> str:
        q = q.strip()
        if len(q) >= 2 and q[0] == q[-1] and q[0] in ("'", '"'):
            return q[1:-1].strip()
        return q

    def _rank_candidates(self, query: str, items: List[Dict]) -> List[Tuple[Dict, float]]:
        q_tokens = self._tokens(query)
        phrase = self._main_phrase(query)
        ranked: List[Tuple[Dict, float]] = []
        for it in items:
            title = (it.get("title") or it.get("source") or "")[:300]
            snippet = (it.get("snippet") or it.get("description") or "")[:600]
            position = self._safe_float(it.get("position"))
            t_overlap = self._overlap(self._tokens(title), q_tokens)
            s_overlap = self._overlap(self._tokens(snippet), q_tokens)
            phrase_hit = 1.0 if (phrase and (phrase in title.lower() or phrase in snippet.lower())) else 0.0
            pos_bonus = (0.4 * (1.0 / math.sqrt(position))) if (position and position > 0) else 0.0
            score = 1.3 * t_overlap + 0.7 * s_overlap + 0.3 * phrase_hit + 0.2 * pos_bonus
            ranked.append((it, float(score)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _dedupe_by_domain_and_title(self, ranked: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        seen = set()
        out: List[Tuple[Dict, float]] = []
        for it, sc in ranked:
            link = it.get("link") or it.get("url") or ""
            domain = self._domain_of(link) or ""
            title = (it.get("title") or it.get("source") or "").strip().lower()
            title = re.sub(r"[\s\-–—|•]+", " ", title)
            key = (domain, title)
            if key in seen:
                continue
            seen.add(key)
            out.append((it, sc))
        return out

    def _normalize_special_answer(self, payload: Dict, kind: str) -> Dict:
        title = payload.get("title") or payload.get("type") or kind.replace("_", " ").title()
        snippet = payload.get("answer") or payload.get("description") or ""
        link = payload.get("link") or payload.get("url") or ""
        return {"title": title, "snippet": snippet, "link": link, "_kind": kind, "position": 0}

    def _tokens(self, s: str) -> List[str]:
        s = (s or "").lower()
        return [t for t in re.findall(r"[a-z0-9]+", s) if t not in self._stop]

    def _overlap(self, a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        A, B = set(a), set(b)
        inter = len(A & B)
        denom = math.sqrt(len(A) * len(B))
        return inter / denom if denom else 0.0

    def _main_phrase(self, q: str) -> Optional[str]:
        m = re.search(r'"([^"]+)"', q or "")
        return m.group(1).lower().strip() if m else None

    def _domain_of(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        try:
            netloc = urlparse(url).netloc.lower()
            return re.sub(r"^(www\d?\.)", "", netloc)
        except Exception:
            return None

    def _safe_float(self, x) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None
