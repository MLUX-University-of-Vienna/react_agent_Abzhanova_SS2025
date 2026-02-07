from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

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


@dataclass
class _HTTPDebug:
    endpoint: str = ""
    method: str = ""
    status: Optional[int] = None
    url: str = ""
    headers: Dict[str, Optional[str]] = field(default_factory=dict)
    body_preview: str = ""
    error: str = ""


class DuckDuckGoTool(AgentTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Web search via DuckDuckGo. Use for real-time or frequently changing facts."

    _REFINER_MAX_CHARS = 300
    _REFINER_INPUT_FORMAT = "just: <query> (defaults to web)\n"
    _REFINER_INSTRUCTIONS = (
        "Return a NORMAL internet search query (plain text words).\n"
        "Rules:\n"
        "- Output MUST be a single-line string (no JSON, no quotes needed).\n"
        "- Prefer 4–12 words; keep the key entity + intent words (e.g., current, now, weather, price).\n"
        "- DO NOT output URL/querystring formats (no key=value, no &, no ?).\n"
        "- DO NOT URL-encode spaces.\n"
        "- Optional recency suffix: time:past1h|past24h|past7d|past30d\n"
        "- If user asks 'right now'/'current', include 'current' or 'now'.\n"
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
        "- Any querystring-like tokens: key=value\n"
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

    DEFAULT_MAX_RESULTS = 10
    DEFAULT_MAX_EVIDENCE_ITEMS = 3
    DEFAULT_EVIDENCE_SNIPPET_MAX_CHARS = 320

    REQUEST_TIMEOUT_SECONDS = 10
    PAGE_PEEK_TIMEOUT_SECONDS = 6
    MAX_PAGE_PEEKS = 2

    USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123 Safari/537.36"
    )

    DEBUG_BODY_PREVIEW_CHARS = 900
    DEBUG_HEADER_KEYS = (
        "content-type",
        "content-length",
        "server",
        "date",
        "retry-after",
        "location",
    )

    _DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"
    _DDG_HTML_URL = "https://html.duckduckgo.com/html/"

    TIME_FILTER_MAP = {
        "past1h": "d",
        "past24h": "d",
        "past7d": "w",
        "past30d": "m",
    }

    _SKIP_DOMAINS_SUFFIX = (
        "facebook.com",
        "instagram.com",
        "tiktok.com",
        "pinterest.com",
        "x.com",
        "twitter.com",
    )

    _MODE_WEB = "web"
    _MODE_NEWS = "news"

    _FX_PATTERNS = (
        re.compile(r"\b1\s*EUR\s*=\s*([0-9]+(?:[.,][0-9]+)?)\s*USD\b", re.I),
        re.compile(r"\b([0-9]+(?:[.,][0-9]+)?)\s*USD\s*(?:for|per)\s*1\s*EUR\b", re.I),
        re.compile(r"\bEUR\s*/\s*USD\b[^0-9]{0,25}([0-9]+(?:[.,][0-9]+)?)", re.I),
        re.compile(r"\bEURUSD\b[^0-9]{0,25}([0-9]+(?:[.,][0-9]+)?)", re.I),
    )

    def __init__(
        self,
        max_results: int = DEFAULT_MAX_RESULTS,
        *,
        max_evidence_items: int = DEFAULT_MAX_EVIDENCE_ITEMS,
        evidence_snippet_max_chars: int = DEFAULT_EVIDENCE_SNIPPET_MAX_CHARS,
        allow_wikipedia: bool = False,
    ):
        super().__init__()
        self.max_results = max(1, int(max_results))
        self.max_evidence_items = max(1, int(max_evidence_items))
        self.evidence_snippet_max_chars = max(80, int(evidence_snippet_max_chars))
        self.allow_wikipedia = bool(allow_wikipedia)

        self._last_http_debug = _HTTPDebug()

        logger.info(
            "[DuckDuckGoTool.__init__] ready max_results=%d max_evidence_items=%d allow_wikipedia=%s",
            self.max_results,
            self.max_evidence_items,
            self.allow_wikipedia,
        )

    def execute(self, tool_input: str) -> Evidence:
        if requests is None:
            return Evidence(
                tool=self.name,
                content="DuckDuckGoTool requires 'requests' but it is not available.",
                url=None,
                extracted={"error": True, "reason": "requests_missing"},
                as_of=datetime.utcnow(),
                confidence=0.1,
            )

        mode, query, df = self._parse_tool_input(tool_input)
        query = (query or "").strip()

        logger.info("[DuckDuckGoTool.execute] mode=%s query=%r df=%r", mode, query, df)

        if not query:
            return Evidence(
                tool=self.name,
                content="Empty query.",
                url=None,
                extracted={"error": True, "reason": "empty_query"},
                as_of=datetime.utcnow(),
                confidence=0.1,
            )

        if BeautifulSoup is None:
            return Evidence(
                tool=self.name,
                content="DuckDuckGoTool requires 'bs4' (BeautifulSoup) but it is not available.",
                url=None,
                extracted={"error": True, "reason": "bs4_missing"},
                as_of=datetime.utcnow(),
                confidence=0.1,
            )

        results, meta = self._search(query=query, df=df)

        if not results:
            dbg = self._last_http_debug
            return Evidence(
                tool=self.name,
                content=(
                    "Search failed: no parseable results from DuckDuckGo.\n"
                    f"HTTP debug: endpoint={dbg.endpoint} method={dbg.method} status={dbg.status} url={dbg.url}\n"
                    f"headers={dbg.headers}\n"
                    f"body_preview={dbg.body_preview}"
                    + (f"\nerror={dbg.error}" if dbg.error else "")
                ),
                url=None,
                extracted={
                    "error": True,
                    "reason": "no_results",
                    "mode": mode,
                    "query": query,
                    "df": df,
                    "endpoint": dbg.endpoint,
                    "method": dbg.method,
                    "status": dbg.status,
                    "url": dbg.url,
                    "headers": dbg.headers,
                    "body_preview": dbg.body_preview,
                    "error_detail": dbg.error,
                    "meta": meta,
                },
                as_of=datetime.utcnow(),
                confidence=0.1,
            )

        # FX extraction from snippets/titles
        fx_candidates = self._extract_fx_from_results(results=results)

        # If likely FX query and no numeric candidate in SERP, peek top pages
        if self._looks_like_fx_query(query) and not fx_candidates:
            fx_candidates = self._peek_top_pages_for_fx(results=results)

        best_fx = self._choose_best_fx_rate(fx_candidates)

        evidence_items: List[Dict[str, str]] = []
        for r in results[: self.max_evidence_items]:
            evidence_items.append(
                {
                    "title": r["title"],
                    "url": r["url"],
                    "snippet": self._truncate(r.get("snippet") or ""),
                }
            )

        primary_url = evidence_items[0]["url"] if evidence_items else None

        content = self._format_evidence_pack(
            mode=mode,
            query=query,
            as_of=self._as_of(),
            items=evidence_items,
            fx_rate=best_fx,
            fx_support_count=len(fx_candidates),
        )

        top_matches: Dict[str, Dict[str, object]] = {}
        for i, r in enumerate(results[: self.max_results], start=1):
            top_matches[str(i)] = {
                "rank": i,
                "title": r["title"],
                "url": r["url"],
                "domain": self._domain_of(r["url"]) or "",
                "snippet": self._truncate(r.get("snippet") or ""),
            }

        extracted = {
            "tool": self.name,
            "mode": mode,
            "query": query,
            "query_used": meta.get("query_used", query),
            "df": df,
            "as_of": self._as_of(),
            "raw_count": len(results),
            "endpoint_used": meta.get("endpoint_used"),
            "http_status": meta.get("status"),
            "top_matches": top_matches,
            "fx_candidates": fx_candidates,
            "fx_best_rate": best_fx,
        }

        confidence = 0.8
        if self._looks_like_fx_query(query):
            confidence = 0.9 if best_fx is not None else 0.55

        return Evidence(
            tool=self.name,
            content=content,
            url=primary_url,
            extracted=extracted,
            as_of=datetime.utcnow(),
            confidence=confidence,
        )

    def _parse_tool_input(self, s: str) -> Tuple[str, str, Optional[str]]:
        s = (s or "").strip()
        m = re.match(r"^(web|news):\s*(.*)$", s, flags=re.I)
        if m:
            mode = m.group(1).lower()
            tail = (m.group(2) or "").strip()
        else:
            mode = self._MODE_WEB
            tail = s

        tail = self._strip_outer_quotes(tail)

        t = re.search(r"\s+time:(past1h|past24h|past7d|past30d)\s*$", tail, flags=re.I)
        df = None
        if t:
            tail = tail[: t.start()].rstrip()
            df = self.TIME_FILTER_MAP.get(t.group(1).lower())

        return mode, tail, df

    def _strip_outer_quotes(self, q: str) -> str:
        q = (q or "").strip()
        if len(q) >= 2 and q[0] == q[-1] and q[0] in ("'", '"'):
            return q[1:-1].strip()
        return q

    def _search(self, *, query: str, df: Optional[str]) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
        session = requests.Session()

        headers = {
            "User-Agent": self.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
            "Connection": "keep-alive",
        }

        queries_to_try = self._query_variants(query)

        last_meta: Dict[str, object] = {}
        for q in queries_to_try:
            html, meta = self._fetch_lite(session=session, headers=headers, query=q, df=df)
            if html and self._looks_like_results(html):
                results = self._parse_any_results(html, query=q)
                if results:
                    return results, {"endpoint_used": "lite", "query_used": q, **meta}

            html2, meta2 = self._fetch_html(session=session, headers=headers, query=q, df=df)
            if html2 and self._looks_like_results(html2):
                results2 = self._parse_any_results(html2, query=q)
                if results2:
                    return results2, {"endpoint_used": "html", "query_used": q, **meta2}

            # forced parse fallback
            if html2:
                results3 = self._parse_any_results(html2, query=q)
                if results3:
                    return results3, {"endpoint_used": "html_forced_parse", "query_used": q, **meta2}
            if html:
                results4 = self._parse_any_results(html, query=q)
                if results4:
                    return results4, {"endpoint_used": "lite_forced_parse", "query_used": q, **meta}

            last_meta = meta2 or meta or {}

        return [], {
            "endpoint_used": last_meta.get("endpoint", ""),
            "query_used": queries_to_try[-1] if queries_to_try else query,
            **last_meta,
        }

    def _query_variants(self, query: str) -> List[str]:
        variants: List[str] = []
        q = self._clean_text(query)

        def add(v: str) -> None:
            vv = self._clean_text(v)
            if vv and vv not in variants:
                variants.append(vv)

        add(q)

        # normalize arrows/symbols that can hurt retrieval
        q_norm = (
            q.replace("→", " to ")
            .replace("->", " to ")
            .replace("—", " ")
            .replace("–", " ")
        )
        add(q_norm)

        # If likely FX query, add one targeted variant
        if self._looks_like_fx_query(q):
            add("1 EUR to USD exchange rate now")

        return variants[:3]

    def _fetch_lite(
        self,
        *,
        session,
        headers: Dict[str, str],
        query: str,
        df: Optional[str],
    ) -> Tuple[Optional[str], Dict[str, object]]:
        params: Dict[str, str] = {"q": query}
        if df:
            params["df"] = df

        return self._http_try(
            session=session,
            method="GET",
            endpoint="lite",
            url=self._DDG_LITE_URL,
            headers=headers,
            params=params,
            data=None,
        )

    def _fetch_html(
        self,
        *,
        session,
        headers: Dict[str, str],
        query: str,
        df: Optional[str],
    ) -> Tuple[Optional[str], Dict[str, object]]:
        data: Dict[str, str] = {"q": query}
        if df:
            data["df"] = df

        headers2 = dict(headers)
        headers2["Referer"] = "https://duckduckgo.com/"

        # Try POST first
        html, meta = self._http_try(
            session=session,
            method="POST",
            endpoint="html",
            url=self._DDG_HTML_URL,
            headers=headers2,
            params=None,
            data=data,
        )
        if html:
            return html, meta

        # Fallback GET
        params = dict(data)
        return self._http_try(
            session=session,
            method="GET",
            endpoint="html_get_fallback",
            url=self._DDG_HTML_URL,
            headers=headers2,
            params=params,
            data=None,
        )

    def _http_try(
        self,
        *,
        session,
        method: str,
        endpoint: str,
        url: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, str]],
        data: Optional[Dict[str, str]],
    ) -> Tuple[Optional[str], Dict[str, object]]:
        """
        Accept 200 and 202 if body looks non-empty.
        Stores last debug into self._last_http_debug.
        """
        last_err = ""
        for attempt in (1, 2, 3):
            try:
                t0 = time.time()
                if method.upper() == "POST":
                    resp = session.post(
                        url,
                        data=data,
                        params=params,
                        headers=headers,
                        timeout=self.REQUEST_TIMEOUT_SECONDS,
                        allow_redirects=True,
                    )
                else:
                    resp = session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=self.REQUEST_TIMEOUT_SECONDS,
                        allow_redirects=True,
                    )

                dt_ms = int((time.time() - t0) * 1000)
                body = resp.text or ""

                hdrs = {k: resp.headers.get(k) for k in self.DEBUG_HEADER_KEYS}
                preview = body[: self.DEBUG_BODY_PREVIEW_CHARS].replace("\n", " ").replace("\r", " ")

                self._last_http_debug = _HTTPDebug(
                    endpoint=endpoint,
                    method=method.upper(),
                    status=resp.status_code,
                    url=resp.url,
                    headers=hdrs,
                    body_preview=preview,
                    error="",
                )

                logger.info(
                    "[DuckDuckGoTool.http] endpoint=%s method=%s attempt=%d status=%d ms=%d",
                    endpoint,
                    method.upper(),
                    attempt,
                    resp.status_code,
                    dt_ms,
                )

                if resp.status_code not in (200, 202):
                    last_err = f"status={resp.status_code}"
                    time.sleep(0.12 * attempt)
                    continue

                if len(body) < 300:
                    last_err = "short_html"
                    time.sleep(0.12 * attempt)
                    continue

                return body, {
                    "endpoint": endpoint,
                    "status": resp.status_code,
                    "url": resp.url,
                    "ms": dt_ms,
                }

            except Exception as e:
                last_err = repr(e)
                self._last_http_debug = _HTTPDebug(
                    endpoint=endpoint,
                    method=method.upper(),
                    status=None,
                    url=url,
                    headers={},
                    body_preview="",
                    error=last_err,
                )
                logger.warning(
                    "[DuckDuckGoTool.http] endpoint=%s attempt=%d failed err=%r",
                    endpoint,
                    attempt,
                    e,
                )
                time.sleep(0.15 * attempt)

        dbg = self._last_http_debug
        dbg.error = last_err
        self._last_http_debug = dbg

        return None, {"endpoint": endpoint, "error": last_err}

    def _looks_like_results(self, html: str) -> bool:
        h = (html or "").lower()
        if not h or len(h) < 500:
            return False

        if "uddg=" in h and h.count("uddg=") >= 3:
            return True
        if "result__a" in h or "result__snippet" in h or "result-link" in h:
            return True
        if h.count("<a ") >= 10 and ("duckduckgo.com/l/?" in h or "/l/?" in h):
            return True
        return False

    def _parse_any_results(self, html: str, *, query: str) -> List[Dict[str, str]]:
        soup = BeautifulSoup(html, "html.parser")
        results: List[Dict[str, str]] = []

        # Pass 1: container-based parsing (robust across DDG variants)
        container_selector = ".result, .results_links, .results_links_deep, .web-result, div.result, tr"
        for c in soup.select(container_selector):
            a = c.select_one(
                "a.result__a[href], a.result-link[href], h2 a[href], a[href*='uddg='], a[href*='/l/?']"
            )
            if not a:
                continue

            href = a.get("href") or ""
            url = self._decode_ddg_redirect(href) or href
            title = self._clean_text(a.get_text(" ", strip=True))
            if not title or not url:
                continue

            sn = c.select_one(".result__snippet, .result-snippet, .snippet, td.result-snippet")
            snippet = self._clean_text(sn.get_text(" ", strip=True)) if sn else ""
            if not snippet:
                snippet = self._find_lite_snippet(a)

            self._maybe_add_result(results, title=title, url=url, snippet=snippet, query=query)

            if len(results) >= self.max_results:
                break

        if results:
            return results

        # Pass 2: broad anchor fallback
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if "uddg=" not in href and "/l/?" not in href:
                continue

            url = self._decode_ddg_redirect(href) or ""
            title = self._clean_text(a.get_text(" ", strip=True))
            if not title or not url:
                continue

            snippet = self._find_lite_snippet(a)
            self._maybe_add_result(results, title=title, url=url, snippet=snippet, query=query)

            if len(results) >= self.max_results:
                break

        return results

    def _find_lite_snippet(self, a) -> str:
        try:
            tr = a.find_parent("tr")
            if tr is None:
                return ""

            # DDG lite often puts snippet in next row(s)
            cur = tr
            for _ in range(2):
                cur = cur.find_next_sibling("tr")
                if cur is None:
                    break
                txt = self._clean_text(cur.get_text(" ", strip=True))
                txt = re.sub(r"\b(cached|similar)\b", "", txt, flags=re.I).strip()
                if txt:
                    return txt
            return ""
        except Exception:
            return ""

    def _maybe_add_result(
        self,
        results: List[Dict[str, str]],
        *,
        title: str,
        url: str,
        snippet: str,
        query: str,
    ) -> None:
        url = (url or "").strip()
        title = (title or "").strip()
        if not url or not title:
            return

        domain = self._domain_of(url) or ""
        ql = (query or "").lower()

        if not self.allow_wikipedia:
            if domain == "wikipedia.org" or domain.endswith(".wikipedia.org"):
                if "wikipedia" not in ql:
                    return

        for bad in self._SKIP_DOMAINS_SUFFIX:
            if domain.endswith(bad):
                return

        for r in results:
            if r.get("url") == url:
                return

        results.append(
            {
                "title": title[:300],
                "url": url,
                "snippet": (snippet or "")[:900],
            }
        )

    def _decode_ddg_redirect(self, href: str) -> Optional[str]:
        href = (href or "").strip()
        if not href:
            return None

        # FIX: protocol-relative URL
        if href.startswith("//"):
            href = "https:" + href

        if href.startswith("http://") or href.startswith("https://"):
            try:
                p = urlparse(href)
                qs = parse_qs(p.query or "")
                uddg = qs.get("uddg", [None])[0]
                if uddg:
                    return unquote(uddg)
            except Exception:
                pass
            return href

        try:
            full = "https://duckduckgo.com" + href if href.startswith("/") else href
            p = urlparse(full)
            qs = parse_qs(p.query or "")
            uddg = qs.get("uddg", [None])[0]
            if uddg:
                return unquote(uddg)
        except Exception:
            return None

        return None

    def _looks_like_fx_query(self, query: str) -> bool:
        q = (query or "").lower()
        return (
            ("eur" in q and "usd" in q)
            or ("euro" in q and "dollar" in q)
            or ("€" in q and "$" in q)
            or ("eurusd" in q)
        )

    def _extract_fx_from_results(self, *, results: List[Dict[str, str]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for r in results[: self.max_results]:
            text = f"{r.get('title', '')} | {r.get('snippet', '')}"
            out.extend(
                self._extract_fx_candidates_from_text(
                    text=text,
                    source_url=r.get("url"),
                    source_kind="snippet",
                )
            )
        return self._dedupe_fx_candidates(out)

    def _peek_top_pages_for_fx(self, *, results: List[Dict[str, str]]) -> List[Dict[str, object]]:
        if requests is None or BeautifulSoup is None:
            return []

        session = requests.Session()
        headers = {
            "User-Agent": self.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

        out: List[Dict[str, object]] = []
        for r in results[: self.MAX_PAGE_PEEKS]:
            url = (r.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue

            try:
                resp = session.get(
                    url,
                    headers=headers,
                    timeout=self.PAGE_PEEK_TIMEOUT_SECONDS,
                    allow_redirects=True,
                )
                if resp.status_code >= 400:
                    continue

                ctype = (resp.headers.get("content-type") or "").lower()
                if "html" not in ctype and "xhtml" not in ctype:
                    continue

                html = resp.text or ""
                if not html:
                    continue

                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                text = self._clean_text(soup.get_text(" ", strip=True))
                if not text:
                    continue

                # limit scanning size
                text = text[:50000]
                out.extend(
                    self._extract_fx_candidates_from_text(
                        text=text,
                        source_url=url,
                        source_kind="page",
                    )
                )
            except Exception:
                continue

        return self._dedupe_fx_candidates(out)

    def _extract_fx_candidates_from_text(
        self,
        *,
        text: str,
        source_url: Optional[str],
        source_kind: str,
    ) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        raw = text or ""
        if not raw:
            return out

        for pat in self._FX_PATTERNS:
            for m in pat.finditer(raw):
                raw_num = (m.group(1) or "").strip()
                val = self._parse_decimal(raw_num)
                if val is None:
                    continue
                if not (0.000001 <= val <= 100000):
                    continue

                out.append(
                    {
                        "pair": "EUR/USD",
                        "value": float(val),
                        "source_kind": source_kind,
                        "source_url": source_url or "",
                        "match": m.group(0)[:160],
                    }
                )
                if len(out) >= 8:
                    return out
        return out

    def _dedupe_fx_candidates(self, cands: List[Dict[str, object]]) -> List[Dict[str, object]]:
        seen = set()
        out: List[Dict[str, object]] = []
        for c in cands:
            try:
                key = (
                    round(float(c.get("value", 0.0)), 6),
                    str(c.get("source_url", "")),
                    str(c.get("source_kind", "")),
                )
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _choose_best_fx_rate(self, cands: List[Dict[str, object]]) -> Optional[float]:
        if not cands:
            return None
        vals = sorted(float(c["value"]) for c in cands if "value" in c)
        if not vals:
            return None
        n = len(vals)
        mid = n // 2
        if n % 2 == 1:
            return round(vals[mid], 6)
        return round((vals[mid - 1] + vals[mid]) / 2.0, 6)

    def _parse_decimal(self, s: str) -> Optional[float]:
        t = (s or "").strip().replace(",", ".")
        try:
            return float(t)
        except Exception:
            return None

    def _as_of(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def _format_evidence_pack(
        self,
        *,
        mode: str,
        query: str,
        as_of: str,
        items: List[Dict[str, str]],
        fx_rate: Optional[float] = None,
        fx_support_count: int = 0,
    ) -> str:
        lines: List[str] = []
        lines.append(f"MODE: {mode}")
        lines.append(f"QUERY: {query}")
        lines.append(f"AS_OF: {as_of}")
        if fx_rate is not None:
            lines.append(f"FX_RATE_EUR_USD: {fx_rate}")
            lines.append(f"FX_RATE_SUPPORT_COUNT: {fx_support_count}")
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
                lines.append(f"ITEM_{i}_INFO: {self._truncate(snippet)}")
            lines.append("")

        return "\n".join(lines).strip()

    def _truncate(self, s: str) -> str:
        s = (s or "").strip()
        if len(s) <= self.evidence_snippet_max_chars:
            return s
        return s[: self.evidence_snippet_max_chars].rstrip() + "..."

    def _clean_text(self, s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def _domain_of(self, url: str) -> Optional[str]:
        if not url:
            return None
        try:
            netloc = urlparse(url).netloc.lower()
            netloc = re.sub(r"^(www\d?\.)", "", netloc)
            return netloc
        except Exception:
            return None
