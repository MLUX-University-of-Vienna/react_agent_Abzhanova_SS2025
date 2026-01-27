from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from serpapi.google_search import GoogleSearch

from my_react_agent.agent_memory.data_structures import Evidence
from my_react_agent.tool_management.tools.agent_tool import AgentTool

logger = logging.getLogger(__name__)

try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


class SerpAPITool(AgentTool):

    DEFAULT_HL = "en"
    DEFAULT_GL: Optional[str] = None
    DEFAULT_MAX_RESULTS = 10
    DEFAULT_NUM_RESULTS = 50

    DEFAULT_PRIMARY_PAGE_MAX_CHARS = 20000

    REQUEST_TIMEOUT_SECONDS = 7
    USER_AGENT = "Mozilla/5.0 (compatible; SerpAPITool/1.0)"

    DEFAULT_EVIDENCE_SNIPPET_MAX_CHARS = 320
    DEFAULT_MAX_EVIDENCE_ITEMS = 3

    DEFAULT_FETCH_PAGE_IF_SNIPPET_EMPTY = False
    DEFAULT_FETCH_EXTRACT_WINDOW_CHARS = 500 

    DEFAULT_LOG_FULL_SERPAPI_JSON = False

    TIME_FILTER_MAP = {
        "past1h": "qdr:h",
        "past24h": "qdr:d",
        "past7d": "qdr:w",
        "past30d": "qdr:m",
    }

    MODE_WEB = "web"
    MODE_NEWS = "news"

    TITLE_OVERLAP_WEIGHT = 1.4
    SNIPPET_OVERLAP_WEIGHT = 0.9
    FOCUS_OVERLAP_WEIGHT = 0.6
    PHRASE_HIT_WEIGHT = 0.4
    POSITION_BONUS_WEIGHT = 0.3
    NUMERIC_INTENT_BONUS = 0.3
    GOOD_DOMAIN_BONUS = 0.15
    BAD_DOMAIN_PENALTY = -0.25

    STOPWORDS = {
        "the", "a", "an", "to", "of", "in", "for", "and", "or", "on", "with", "by", "is", "are", "at", "from", "as",
        "that", "this", "these", "those", "be", "was", "were", "it", "its", "about", "how", "what", "when", "where",
        "who", "why", "which", "can", "i", "you",
    }

    QUANT_TOKENS = {
        "number", "count", "total", "sum",
        "population", "inhabitants", "einwohner", "residents", "people",
        "area", "size", "distance", "height", "length", "weight",
        "age", "price", "cost", "salary", "revenue", "turnover", "gdp",
        "ratio", "percentage", "percent", "rate", "score", "ranking",
        "results", "statistic", "statistics", "data",
    }

    GOOD_DOMAIN_PATTERNS = [r"\.gov(\.|$)", r"\.edu(\.|$)", r"\.ac\.", r"\.org$"]
    BAD_DOMAIN_PATTERNS = [
        r"facebook\.com$", r"instagram\.com$", r"tiktok\.com$", r"(?:^|\.)(twitter|x)\.com$", r"pinterest\.com$"
    ]

    _FALLBACK_NUM_PAT = re.compile(r"\d")

    _REFINER_MAX_CHARS = 300

    _REFINER_INPUT_FORMAT = (
        'web: <query> [time:past1h|past24h|past7d|past30d]\n'
        'news: <query> [time:past1h|past24h|past7d|past30d]\n'
        'or just: <query>    (defaults to web)\n'
    )

    _REFINER_INSTRUCTIONS = (
        "Return a NORMAL internet search query (plain text words).\n"
        "Rules:\n"
        "- Output MUST be a single-line string (no JSON, no quotes needed).\n"
        "- Prefer 4–12 words; keep the key entity + intent words (e.g., current, now, weather, price).\n"
        "- DO NOT output URL/querystring formats (no key=value, no &, no ?).\n"
        "- DO NOT URL-encode spaces (avoid replacing spaces with + or %20).\n"
        "- Only add a time filter if it helps recency, using the allowed suffix exactly: time:past1h|past24h|past7d|past30d\n"
        "- If user asks 'right now'/'current', include the plain word 'current' or 'now'.\n"
        "Good: Tokyo Japan current temperature now\n"
        "Bad: temperature+Tokyo,+Japan&timestamp=now\n"
    )

    _REFINER_INPUT_REGEX = (
        r"^(?:(?:web|news):\s*)?"
        r"[^{}\n\r<>]{1,300}"
        r"(?:\s+time:(?:past1h|past24h|past7d|past30d))?\s*$"
    )

    _REFINER_FORBIDDEN = (
        "Forbidden patterns / tokens:\n"
        "- Any querystring-like tokens: key=value (e.g., timestamp=now, q=tokyo)\n"
        "- Characters: &, ?, =\n"
        "- URL encoding: %20, %2F, etc.\n"
        "- Replacing spaces with '+' (unless it is part of a legitimate term like C++)\n"
    )


    @property
    def refiner_instructions(self) -> str:
        return self._REFINER_INSTRUCTIONS

    @property
    def refiner_input_format(self) -> str:
        return self._REFINER_INPUT_FORMAT

    @property
    def refiner_input_regex(self) -> Optional[str]:
        return self._REFINER_INPUT_REGEX

    @property
    def refiner_forbidden(self) -> str:
        return self._REFINER_FORBIDDEN


    @property
    def refiner_max_chars(self) -> int:
        return int(self._REFINER_MAX_CHARS)


    def __init__(
        self,
        api_key: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        hl: str = DEFAULT_HL,
        gl: Optional[str] = DEFAULT_GL,
        num_results: int = DEFAULT_NUM_RESULTS,
        primary_page_max_chars: int = DEFAULT_PRIMARY_PAGE_MAX_CHARS,
        *,
        evidence_snippet_max_chars: int = DEFAULT_EVIDENCE_SNIPPET_MAX_CHARS,
        max_evidence_items: int = DEFAULT_MAX_EVIDENCE_ITEMS,
        fetch_page_if_snippet_empty: bool = DEFAULT_FETCH_PAGE_IF_SNIPPET_EMPTY,
        fetch_extract_window_chars: int = DEFAULT_FETCH_EXTRACT_WINDOW_CHARS,
        log_full_serpapi_json: bool = DEFAULT_LOG_FULL_SERPAPI_JSON,
    ):
        self.api_key = api_key
        self.max_results = int(max_results)
        self.hl = hl
        self.gl = gl
        self.num_results = int(num_results)
        self.primary_page_max_chars = int(primary_page_max_chars)

        self.evidence_snippet_max_chars = max(50, int(evidence_snippet_max_chars))
        self.max_evidence_items = max(1, int(max_evidence_items))
        self.fetch_page_if_snippet_empty = bool(fetch_page_if_snippet_empty)
        self.fetch_extract_window_chars = max(200, int(fetch_extract_window_chars))

        self.log_full_serpapi_json = bool(log_full_serpapi_json)

        self._stop = self.STOPWORDS
        self._quant_tokens = self.QUANT_TOKENS
        self._good_domain_patterns = self.GOOD_DOMAIN_PATTERNS
        self._bad_domain_patterns = self.BAD_DOMAIN_PATTERNS

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Web search via SerpAPI (Google). Use for real-time or frequently changing facts "

    def execute(self, tool_input: str) -> Evidence:
        try:
            mode, query, tbs = self._parse_tool_input(tool_input)
            params = self._build_search_params(mode, query, tbs)
            data = GoogleSearch(params).get_dict()

            logger.debug("[SerpAPITool.execute] Raw SerpAPI response keys=%s", list(data.keys()))
            if self.log_full_serpapi_json:
                logger.debug("[SerpAPITool.execute] Raw SerpAPI response FULL:\n%s", data)

            candidates = self._collect_candidates(mode, data)
            if not candidates:
                content = self._format_evidence_pack(
                    query=query,
                    as_of=self._serp_as_of(data),
                    items=[],
                    mode=mode,
                )
                logger.info("[SerpAPITool.execute] Evidence pack:\n%s", (content or "").strip())
                return self._evidence_ok(
                    content=content or "No results found.",
                    mode=mode,
                    query=query,
                    url=None,
                    extracted={
                        "reason": "no_candidates",
                        "serpapi_error": str(data.get("error") or ""),
                        "raw_count": 0,
                        "as_of": self._serp_as_of(data) or "",
                    },
                    confidence=0.7,
                )

            ranked = self._rank_candidates(query, candidates)
            deduped = self._dedupe_by_domain_and_title(ranked)
            filtered = self._filter_wikipedia(deduped)

            if not filtered:
                content = self._format_evidence_pack(
                    query=query,
                    as_of=self._serp_as_of(data),
                    items=[],
                    mode=mode,
                )
                logger.info("[SerpAPITool.execute] Evidence pack:\n%s", content)
                return self._evidence_ok(
                    content=content or "Search returned mainly Wikipedia results or no suitable non-Wikipedia pages.",
                    mode=mode,
                    query=query,
                    url=None,
                    extracted={
                        "reason": "only_wikipedia_or_filtered_out",
                        "raw_count": len(candidates),
                        "as_of": self._serp_as_of(data) or "",
                    },
                    confidence=0.7,
                )

            evidence_items = self._select_evidence_items(
                query=query,
                data=data,
                filtered_ranked=filtered,
                mode=mode,
            )

            primary_url = ""
            for it in evidence_items:
                if it.get("url"):
                    primary_url = str(it["url"])
                    break
            if not primary_url:
                top_it = filtered[0][0]
                primary_url = (top_it.get("link") or top_it.get("url") or "") or ""

            primary_domain = self._domain_of(primary_url) if primary_url else ""
            primary_title = ""
            if filtered:
                top_it = filtered[0][0]
                primary_title = (top_it.get("title") or top_it.get("source") or "") or ""

            content = self._format_evidence_pack(
                query=query,
                as_of=self._serp_as_of(data),
                items=evidence_items,
                mode=mode,
            )

            logger.info("[SerpAPITool.execute] Evidence pack:\n%s", content)

            top_matches: Dict[str, Dict[str, object]] = {}
            for rank, (it, sc) in enumerate(filtered[: self.max_results], start=1):
                link = it.get("link") or it.get("url") or ""
                top_matches[str(rank)] = {
                    "rank": rank,
                    "score": round(float(sc), 3),
                    "title": (it.get("title") or it.get("source") or ""),
                    "snippet": (it.get("snippet") or it.get("description") or "")[: self.evidence_snippet_max_chars],
                    "url": link,
                    "domain": self._domain_of(link) or "",
                    "position": int(it.get("position") or 0),
                    "kind": str(it.get("_kind") or ""),
                }

            extracted = {
                "tool": self.name,
                "mode": mode,
                "query": query,
                "as_of": self._serp_as_of(data) or "",
                "primary_title": primary_title,
                "primary_domain": primary_domain or "",
                "raw_count": len(candidates),
                "top_matches": top_matches,
            }

            return Evidence(
                tool=self.name,
                content=(content or "").strip(),
                url=primary_url or None,
                extracted=extracted,
                as_of=datetime.utcnow(),
                confidence=0.8,
            )

        except Exception as e:
            logger.exception("[SerpAPITool.execute] search_failed")
            return Evidence(
                tool=self.name,
                content=f"Search failed: {str(e)}",
                url=None,
                extracted={"error": True, "exception": repr(e)},
                as_of=datetime.utcnow(),
                confidence=0.1,
            )

    def _serp_as_of(self, data: Dict) -> str:
        """
        Prefer SerpAPI's created_at (UTC string). Fall back to now-UTC ISO-like.
        """
        md = (data or {}).get("search_metadata", {}) or {}
        created_at = md.get("created_at")
        if isinstance(created_at, str) and created_at.strip():
            return created_at.strip()
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def _build_search_params(self, mode: str, query: str, tbs: Optional[str]) -> Dict[str, object]:
        params: Dict[str, object] = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "hl": self.hl,
            "num": self.num_results,
        }
        if self.gl:
            params["gl"] = self.gl
        if mode == self.MODE_NEWS:
            params["tbm"] = "nws"
        if tbs:
            params["tbs"] = tbs
        return params

    def _collect_candidates(self, mode: str, data: Dict) -> List[Dict]:
        if mode == self.MODE_NEWS:
            return data.get("news_results", []) or []

        candidates: List[Dict] = []
        answer_box = data.get("answer_box")
        knowledge_graph = data.get("knowledge_graph")
        ai_overview = data.get("ai_overview")
        organic_results = data.get("organic_results", []) or []

        if answer_box:
            candidates.append(self._normalize_special_answer(answer_box, "answer_box"))
        if knowledge_graph:
            candidates.append(self._normalize_special_answer(knowledge_graph, "knowledge_graph"))
        if ai_overview:
            candidates.append(self._normalize_ai_overview(ai_overview))

        candidates.extend(organic_results)
        return candidates

    def _parse_tool_input(self, s: str) -> Tuple[str, str, Optional[str]]:
        s = (s or "").strip()
        m = re.match(r"^(web|news):\s*(.*)$", s, flags=re.I)
        if not m:
            return self.MODE_WEB, self._strip_outer_quotes(s), None
        mode = m.group(1).lower()
        tail = m.group(2).strip()
        t = re.search(r"\s+time:(past1h|past24h|past7d|past30d)\s*$", tail, flags=re.I)
        tbs = None
        if t:
            tail = tail[: t.start()].rstrip()
            tbs = self.TIME_FILTER_MAP[t.group(1).lower()]
        return mode, self._strip_outer_quotes(tail), tbs

    def _strip_outer_quotes(self, q: str) -> str:
        q = q.strip()
        if len(q) >= 2 and q[0] == q[-1] and q[0] in ("'", '"'):
            return q[1:-1].strip()
        return q

    def _normalize_special_answer(self, payload: Dict, kind: str) -> Dict:
        title = payload.get("title") or payload.get("type") or kind.replace("_", " ").title()

        snippet_parts: List[str] = []
        if payload.get("answer"):
            snippet_parts.append(str(payload["answer"]))
        if payload.get("snippet") and payload["snippet"] not in snippet_parts:
            snippet_parts.append(str(payload["snippet"]))
        if payload.get("description") and payload["description"] not in snippet_parts:
            snippet_parts.append(str(payload["description"]))

        snippet = " | ".join([p for p in snippet_parts if p]).strip()
        link = payload.get("link") or payload.get("url") or payload.get("website") or ""

        return {
            "title": title,
            "snippet": snippet,
            "link": link,
            "_kind": kind,
            "position": 0,
            "_payload": payload,
        }

    def _normalize_ai_overview(self, payload: Dict) -> Dict:
        text_blocks = payload.get("text_blocks") or []
        para = ""
        if isinstance(text_blocks, list):
            for b in text_blocks:
                if isinstance(b, dict) and b.get("type") == "paragraph":
                    para = str(b.get("snippet") or "").strip()
                    if para:
                        break

        refs = payload.get("references") or []
        ref_lines: List[str] = []
        if isinstance(refs, list):
            for r in refs[:2]:
                if not isinstance(r, dict):
                    continue
                title = (r.get("title") or r.get("source") or "").strip()
                link = (r.get("link") or "").strip()
                if title and link:
                    ref_lines.append(f"{title} ({link})")

        snippet = para
        if ref_lines:
            snippet = (snippet + " | Sources: " + "; ".join(ref_lines)).strip()

        return {
            "title": "AI overview",
            "snippet": snippet,
            "link": "",
            "_kind": "ai_overview",
            "position": 0,
            "_payload": payload,
        }

    def _select_evidence_items(
        self,
        *,
        query: str,
        data: Dict,
        filtered_ranked: List[Tuple[Dict, float]],
        mode: str,
    ) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []

        for key in ("answer_box", "knowledge_graph", "ai_overview"):
            c = self._find_candidate_by_kind(filtered_ranked, key)
            if c:
                title, url, snippet = self._candidate_fields(c)
                snippet = self._clean_snippet(snippet)
                if snippet:
                    out.append({"title": title, "url": url, "snippet": self._truncate(snippet)})
            if len(out) >= max(1, self.max_evidence_items // 2):
                break

        for (it, _sc) in filtered_ranked:
            if len(out) >= self.max_evidence_items:
                break
            title, url, snippet = self._candidate_fields(it)
            snippet = self._clean_snippet(snippet)
            url = (url or "").strip()

            if not snippet and url and self.fetch_page_if_snippet_empty:
                page_text = self._fetch_page_text(url, max_chars=self.primary_page_max_chars)
                if page_text:
                    snippet = self._extract_relevant_window(query=query, text=page_text) or ""

            if not title and not snippet:
                continue

            if self._already_in(out, url=url, title=title):
                continue

            out.append({"title": title, "url": url, "snippet": self._truncate(snippet)})

        return out[: self.max_evidence_items]

    def _find_candidate_by_kind(self, ranked: List[Tuple[Dict, float]], kind: str) -> Optional[Dict]:
        for it, _sc in ranked:
            if str(it.get("_kind") or "").strip().lower() == kind.lower():
                return it
        return None

    def _candidate_fields(self, it: Dict) -> Tuple[str, str, str]:
        title = (it.get("title") or it.get("source") or "").strip()
        snippet = (it.get("snippet") or it.get("description") or "").strip()
        url = (it.get("link") or it.get("url") or "").strip()
        return title, url, snippet

    def _already_in(self, items: List[Dict[str, str]], *, url: str, title: str) -> bool:
        url = (url or "").strip()
        title_key = re.sub(r"\s+", " ", (title or "").strip().lower())
        domain = self._domain_of(url) or ""

        for it in items:
            if url and it.get("url") == url:
                return True
            if domain and title_key:
                other_domain = self._domain_of(it.get("url") or "") or ""
                other_title_key = re.sub(r"\s+", " ", (it.get("title") or "").strip().lower())
                if domain == other_domain and title_key == other_title_key:
                    return True
        return False

    def _format_evidence_pack(
        self,
        *,
        query: str,
        as_of: str,
        items: List[Dict[str, str]],
        mode: str,
    ) -> str:
        lines: List[str] = []
        lines.append(f"MODE: {mode}")
        lines.append(f"QUERY: {query}")
        lines.append(f"AS_OF: {as_of}")
        lines.append("")

        if not items:
            lines.append("NO_EVIDENCE: true")
            return "\n".join(lines).strip()

        for i, it in enumerate(items, start=1):
            title = (it.get("title") or "").strip()
            url = (it.get("url") or "").strip()
            dom = self._domain_of(url) or ""
            snippet = (it.get("snippet") or "").strip()

            lines.append(f"ITEM_{i}_TITLE: {title}")
            lines.append(f"ITEM_{i}_SOURCE: {url}")
            if dom:
                lines.append(f"ITEM_{i}_DOMAIN: {dom}")
            if snippet:
                lines.append(f"ITEM_{i}_INFO: {snippet}")
            lines.append("")

        return "\n".join(lines).strip()

    def _clean_snippet(self, s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def _truncate(self, s: str) -> str:
        s = (s or "").strip()
        if len(s) <= self.evidence_snippet_max_chars:
            return s
        return s[: self.evidence_snippet_max_chars].rstrip() + "..."

    def _extract_relevant_window(self, *, query: str, text: str) -> str:
        q_tokens = self._tokens(query)
        if not text:
            return ""
        t = re.sub(r"\s+", " ", text).strip()
        if not t:
            return ""

        hit_idx = -1
        lower = t.lower()
        for tok in q_tokens[:12]:
            if len(tok) < 3:
                continue
            idx = lower.find(tok.lower())
            if idx >= 0:
                hit_idx = idx
                break

        win = self.fetch_extract_window_chars
        if hit_idx >= 0:
            start = max(0, hit_idx - win // 2)
            end = min(len(t), start + win)
            chunk = t[start:end].strip()
            if start > 0:
                chunk = "..." + chunk
            if end < len(t):
                chunk = chunk + "..."
            return f"{chunk}"
        return (t[:win].rstrip() + ("..." if len(t) > win else ""))


    def _fetch_page_text(self, url: Optional[str], max_chars: int) -> Optional[str]:
        if not url or requests is None:
            return None
        try:
            resp = requests.get(url, timeout=self.REQUEST_TIMEOUT_SECONDS, headers={"User-Agent": self.USER_AGENT})
            if resp.status_code != 200:
                return None
            text = resp.text
            if BeautifulSoup is not None:
                soup = BeautifulSoup(text, "html.parser")
                for bad in soup(["script", "style", "noscript"]):
                    bad.extract()
                text = soup.get_text(separator="\n")
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars] if max_chars and len(text) > max_chars else text
        except Exception:
            return None

    def _tokens(self, s: str) -> List[str]:
        return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if t not in self._stop]

    def _overlap(self, a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        A, B = set(a), set(b)
        inter = len(A & B)
        denom = math.sqrt(len(A) * len(B))
        return inter / denom if denom else 0.0

    def _main_phrase(self, q: str) -> Optional[str]:
        m = re.search(r'"([^"]+)"', q or "")
        return m.group(1).strip() if m else None

    def _domain_of(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        try:
            netloc = urlparse(url).netloc.lower()
            netloc = re.sub(r"^(www\d?\.)", "", netloc)
            return netloc
        except Exception:
            return None

    def _safe_float(self, x) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    def _has_quant_intent(self, q: str) -> bool:
        tokens = re.findall(r"[a-z0-9]+", (q or "").lower())
        return any(tok in self._quant_tokens for tok in tokens)

    def _has_any_number(self, s: str) -> bool:
        return bool(re.search(r"\d", s or ""))

    def _domain_quality(self, domain: str) -> float:
        if not domain or domain == "wikipedia.org":
            return 0.0
        score = 0.0
        for pat in self._good_domain_patterns:
            if re.search(pat, domain):
                score += self.GOOD_DOMAIN_BONUS
        for pat in self._bad_domain_patterns:
            if re.search(pat, domain):
                score += self.BAD_DOMAIN_PENALTY
        return score

    def _rank_candidates(self, query: str, items: List[Dict]) -> List[Tuple[Dict, float]]:
        query_l = (query or "").lower()
        q_tokens = self._tokens(query_l)
        phrase = self._main_phrase(query_l)
        focus_tokens = set(q_tokens)
        quant_intent = self._has_quant_intent(query_l)

        ranked: List[Tuple[Dict, float]] = []
        for it in items:
            title = (it.get("title") or it.get("source") or "")[:300]
            snippet = (it.get("snippet") or it.get("description") or "")[:800]
            position = self._safe_float(it.get("position"))
            link = it.get("link") or it.get("url") or ""
            domain = self._domain_of(link) or ""

            title_tokens = self._tokens(title)
            snippet_tokens = self._tokens(snippet)
            all_tokens = set(title_tokens) | set(snippet_tokens)

            t_overlap = self._overlap(title_tokens, q_tokens)
            s_overlap = self._overlap(snippet_tokens, q_tokens)

            focus_overlap = 0.0
            if focus_tokens:
                inter = all_tokens & focus_tokens
                focus_overlap = len(inter) / max(1.0, float(len(focus_tokens)))

            phrase_hit = 0.0
            if phrase:
                phrase_l = phrase.lower()
                if phrase_l in title.lower() or phrase_l in snippet.lower():
                    phrase_hit = 1.0

            pos_bonus = 1.0 / math.sqrt(position) if position and position > 0 else 0.0
            numeric_bonus = (
                self.NUMERIC_INTENT_BONUS
                if quant_intent and self._has_any_number(title + " " + snippet)
                else 0.0
            )
            domain_bonus = self._domain_quality(domain)

            base_score = (
                self.TITLE_OVERLAP_WEIGHT * t_overlap
                + self.SNIPPET_OVERLAP_WEIGHT * s_overlap
                + self.FOCUS_OVERLAP_WEIGHT * focus_overlap
                + self.PHRASE_HIT_WEIGHT * phrase_hit
                + self.POSITION_BONUS_WEIGHT * pos_bonus
            )
            ranked.append((it, float(base_score + numeric_bonus + domain_bonus)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _dedupe_by_domain_and_title(self, ranked: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        seen = set()
        out: List[Tuple[Dict, float]] = []
        for it, sc in ranked:
            link = it.get("link") or it.get("url") or ""
            domain = self._domain_of(link) or ""
            title = re.sub(r"[\s\-–—|•]+", " ", (it.get("title") or it.get("source") or "").strip().lower())
            key = (domain, title)
            if key in seen:
                continue
            seen.add(key)
            out.append((it, sc))
        return out

    def _filter_wikipedia(self, ranked: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        out: List[Tuple[Dict, float]] = []
        for it, score in ranked:
            link = it.get("link") or it.get("url") or ""
            domain = self._domain_of(link) or ""
            if domain == "wikipedia.org":
                continue
            out.append((it, score))
        return out

    def _evidence_ok(
        self,
        *,
        content: str,
        mode: str,
        query: str,
        url: Optional[str],
        extracted: Dict[str, object],
        confidence: float,
    ) -> Evidence:
        base = {"tool": self.name, "mode": mode, "query": query}
        base.update(extracted or {})
        return Evidence(
            tool=self.name,
            content=content,
            url=url,
            extracted=base,
            as_of=datetime.utcnow(),
            confidence=float(confidence),
        )
