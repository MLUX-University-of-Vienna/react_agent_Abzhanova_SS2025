from __future__ import annotations

import html as html_lib
import logging
import re
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import wikipedia
import wikipediaapi
from bs4 import BeautifulSoup

from my_react_agent.agent_memory.data_structures import Evidence
from my_react_agent.tool_management.tools.agent_tool import AgentTool

logger = logging.getLogger(__name__)


class WikipediaTool(AgentTool):

    LANGUAGE = "en"
    WIKI_BASE_URL = "https://en.wikipedia.org/wiki/"
    DEFAULT_SEARCH_RESULTS = 10
    DEFAULT_MAX_PAGES = 3
    MAX_TITLES = 3
    MAX_DISAMBIG_TEXT_CHARS = 600

    SCORE_PARTS_WEIGHT = 0.5
    SCORE_TOKENS_WEIGHT = 0.3
    SCORE_PREFIX_WEIGHT = 0.1
    SCORE_COMMA_BONUS = 0.1

    CONTENT_PREVIEW_LEN = 6000

    _TITLE_SPLIT_RX = re.compile(r"\s*\|\s*")
    _PLACEHOLDER_RX = re.compile(r"\[[^\]]+\]|\{[^}]+\}", re.I)
    _JOINED_PROMPT_RX = re.compile(r"\s(?:\+| and |,)\s", re.I)
    _STRIP_PARENS_RX = re.compile(r"\((.*?)\)")
    _MAY_REFER_TO_RX = re.compile(r"\bmay refer to:\b", re.I)
    _TAIL_OF_PREP_RX = re.compile(r".*\b(?:of|in|for|about|on|at|from|near)\s+(.+)$", re.I)

    _CITATION_RX = re.compile(r"\s*\[[^\]]+\]\s*")
    _BULLET_RX = re.compile(r"^\s*[•·]\s*")
    _SPACES_RX = re.compile(r"\s{2,}")

    def __init__(
        self,
        user_agent: str = "AIResearchBot",
        max_content_length: int = 10000,
        prose_chars_target: int = 5500,
        infobox_preview_max_chars: int = 4500,
        infobox_rows_preview_limit: int = 80,
        include_infobox_kv_in_extracted: bool = True,
    ):
        self.wiki_api = wikipediaapi.Wikipedia(
            language=self.LANGUAGE,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent=user_agent,
        )

        self.max_content_length = int(max_content_length)
        self.search_results = self.DEFAULT_SEARCH_RESULTS
        self.max_pages = self.DEFAULT_MAX_PAGES

        self.prose_chars_target = int(prose_chars_target)
        self.infobox_preview_max_chars = int(infobox_preview_max_chars)
        self.infobox_rows_preview_limit = int(infobox_rows_preview_limit)
        self.include_infobox_kv_in_extracted = bool(include_infobox_kv_in_extracted)

    @property
    def name(self) -> str:
        return "wikipedia"

    @property
    def description(self) -> str:
        return "Wikipedia lookup by exact page title. Use for stable background facts. Not reliable for real-time data."

    @property
    def refiner_instructions(self) -> str:
        return "Output ' <Exact Wikipedia page title>'. No prose. Write full entity to avoid pages with similar/same names or synonyms"

    @property
    def refiner_input_format(self) -> str:
        return "<Exact Page Title>"

    @property
    def refiner_input_regex(self) -> Optional[str]:
        return r"^(?:\s*)?[^\n\r{}<>]{1,200}(?:\s*\|\s*[^\n\r{}<>]{1,200}){0,2}$"

    @property
    def refiner_forbidden(self) -> str:
        return "questions, multi_line, json, placeholders, plus_sign, wikipedia_prefix"

    @property
    def refiner_examples(self) -> str:
        return "\n".join(["Astana, Kazakhstan", "Madonna", "Python (programming language)"])

    def execute(self, tool_input: str) -> Evidence:
        raw = (tool_input or "").strip()
        logger.info("[Wikipedia.execute] input=%r", raw)
        if not raw:
            evidence = self._evidence_error("No title provided.", reason="empty_input")
            self._log_tool_result(evidence, raw_input=raw, outcome="error")
            return evidence

        candidates = self._parse_titles_or_coerce(raw)
        if not candidates:
            evidence = self._evidence_error(
                "Invalid Wikipedia title input. Provide an exact title (optionally multiple with '|').",
                reason="invalid_format",
                extra={"input": raw},
            )
            self._log_tool_result(evidence, raw_input=raw, outcome="error")
            return evidence

        for candidate_title in candidates[: self.max_pages]:
            page, resolved_via = self._resolve_page(candidate_title)
            if page and page.exists() and not self._is_disambiguation(page):
                evidence = self._evidence_from_page(page, subject=candidate_title, resolved_via=resolved_via)
                self._log_tool_result(
                    evidence,
                    raw_input=raw,
                    outcome="ok",
                    candidate_title=candidate_title,
                    resolved_via=resolved_via,
                )
                return evidence

        evidence = self._evidence_ok(
            content=f"No page found for any of: {', '.join(map(repr, candidates))}.",
            url=None,
            extracted={"reason": "not_found", "candidates": self._dictify_list_of_strings(candidates)},
            confidence=0.7,
        )
        self._log_tool_result(evidence, raw_input=raw, outcome="not_found")
        return evidence

    def _log_tool_result(
        self,
        evidence: Evidence,
        *,
        raw_input: str,
        outcome: str,
        candidate_title: Optional[str] = None,
        resolved_via: Optional[str] = None,
    ) -> None:
        try:
            evidence_url = getattr(evidence, "url", None)
            evidence_confidence = getattr(evidence, "confidence", None)
            evidence_content = getattr(evidence, "content", "") or ""
            evidence_extracted = getattr(evidence, "extracted", {}) or {}

            page_title = ""
            try:
                page_title = str(evidence_extracted.get("page_title") or "")
            except Exception:
                page_title = ""

            logger.info(
                "[Wikipedia.execute] result outcome=ok input_len=38 content_len=5964 confidence=0.85 url='https://en.wikipedia.org/wiki/Burgau' page_title='Burgau' candidate='Burgau' via='exact' extracted_keys=11"
                if False
                else "[Wikipedia.execute] result outcome=%s input_len=%d content_len=%d confidence=%s url=%r page_title=%r candidate=%r via=%r extracted_keys=%d",
                outcome,
                len(raw_input or ""),
                len(evidence_content),
                evidence_confidence,
                evidence_url,
                page_title,
                candidate_title,
                resolved_via,
                len(evidence_extracted.keys()) if isinstance(evidence_extracted, dict) else 0,
            )

            logger.debug("[Wikipedia.execute] result_content:\n%s", evidence_content)

        except Exception as e:
            logger.warning("[Wikipedia.execute] result logging failed error=%r", e)



    def _parse_titles_or_coerce(self, q: str) -> List[str]:
        m = re.match(r"^title:\s*(.+)$", q, flags=re.I)
        s = m.group(1).strip() if m else q

        if self._PLACEHOLDER_RX.search(s):
            return []
        if self._JOINED_PROMPT_RX.search(s):
            return []

        if "|" in s:
            parts = [p.strip() for p in self._TITLE_SPLIT_RX.split(s) if p.strip()]
            return parts[: self.MAX_TITLES]

        candidate, _ = self._coerce_candidate_to_title(s)
        return [candidate] if candidate else []

    def _coerce_candidate_to_title(self, text: str) -> Tuple[Optional[str], str]:
        s = (text or "").strip()
        original = s

        if s.lower().startswith("wikipedia:"):
            s = s.split(":", 1)[1].strip()

        s = self._STRIP_PARENS_RX.sub("", s).strip()

        m = self._TAIL_OF_PREP_RX.match(s)
        if m:
            s = m.group(1).strip()

        s = re.sub(r"^(?:the\s+)?(?:current|latest|total)\s+", "", s, flags=re.I).strip()

        if s and "|" not in s:
            return s, "coerced_from_query"
        return None, f"unusable_after_coercion(original={original!r})"

    def _score_search_result(self, query_title: str, candidate_title: str) -> float:
        q = (query_title or "").strip().lower()
        c = (candidate_title or "").strip().lower()
        if not c:
            return 0.0
        if q and c == q:
            return 1.0

        score = 0.0
        q_parts = [p.strip() for p in q.split(",") if p.strip()]
        c_parts = [p.strip() for p in c.split(",") if p.strip()]
        if q_parts and c_parts:
            score += self.SCORE_PARTS_WEIGHT * (len(set(q_parts) & set(c_parts)) / max(1, len(q_parts)))

        q_tokens = set(re.findall(r"\w+", q))
        c_tokens = set(re.findall(r"\w+", c))
        if q_tokens:
            score += self.SCORE_TOKENS_WEIGHT * (len(q_tokens & c_tokens) / max(1, len(q_tokens)))

        if q.startswith(c) or c.startswith(q):
            score += self.SCORE_PREFIX_WEIGHT
        if "," in q and "," in c:
            score += self.SCORE_COMMA_BONUS

        return max(0.0, min(1.0, score))

    def _resolve_page(self, title: str) -> Tuple[Optional[wikipediaapi.WikipediaPage], str]:
        try:
            page = self.wiki_api.page(title)
        except Exception:
            page = None

        if page and page.exists() and not self._is_disambiguation(page):
            return page, "exact"

        try:
            suggestion = wikipedia.suggest(title)
        except Exception:
            suggestion = None

        if suggestion:
            try:
                pg = self.wiki_api.page(suggestion)
            except Exception:
                pg = None
            if pg and pg.exists() and not self._is_disambiguation(pg):
                return pg, "suggest"

        try:
            raw_results = wikipedia.search(title, results=self.search_results)
        except Exception:
            raw_results = []

        results = sorted(raw_results or [], key=lambda r: self._score_search_result(title, r), reverse=True)

        for r in results:
            if not r or "disambiguation" in r.lower():
                continue
            try:
                pg = self.wiki_api.page(r)
            except Exception:
                continue
            if pg and pg.exists() and not self._is_disambiguation(pg):
                return pg, "search"

        return None, "unresolved"

    def _is_disambiguation(self, page: wikipediaapi.WikipediaPage) -> bool:
        try:
            cats = getattr(page, "categories", {}) or {}
            if any("disambiguation" in k.lower() for k in cats.keys()):
                return True
        except Exception:
            pass

        try:
            txt = (page.summary or page.text or "")[: self.MAX_DISAMBIG_TEXT_CHARS].lower()
            if self._MAY_REFER_TO_RX.search(txt):
                return True
        except Exception:
            pass

        return False

    def _clean_text(self, s: str) -> str:
        if not s:
            return ""
        s = html_lib.unescape(s)
        s = s.replace("\xa0", " ")
        s = self._CITATION_RX.sub(" ", s)
        s = self._SPACES_RX.sub(" ", s).strip()
        return s

    def _clean_label(self, s: str) -> str:
        s = self._clean_text(s)
        s = self._BULLET_RX.sub("", s)
        s = s.replace("•", "").strip()
        return s

    def _clean_value(self, s: str) -> str:
        s = self._clean_text(s)
        s = s.replace("km 2", "km²").replace("m 2", "m²").replace("/km 2", "/km²")
        s = s.replace("( CET )", "(CET)").replace("( CEST )", "(CEST)").replace("( DST )", "(DST)")
        s = self._SPACES_RX.sub(" ", s).strip()
        return s

    def _fetch_page_html(self, page_title: str) -> str:
        try:
            wp_page = wikipedia.page(page_title, auto_suggest=False)
            return wp_page.html() or ""
        except Exception as e:
            logger.warning("[Wikipedia._fetch_page_html] failed title=%r error=%r", page_title, e)
            return ""

    def _extract_infobox_grouped(self, page_title: str) -> Dict[str, Any]:
        html = self._fetch_page_html(page_title)
        if not html:
            return {}

        soup = BeautifulSoup(html, "html.parser")
        table = soup.select_one("table.infobox")
        if not table:
            return {}

        def clean_node_text(node) -> str:
            if not node:
                return ""
            for sup in node.select("sup.reference"):
                sup.decompose()
            for bad in node.select("script, style"):
                bad.decompose()
            return self._clean_text(node.get_text(" ", strip=True))

        infobox_title = self._clean_text(clean_node_text(table.find("caption")))
        image_caption = self._clean_text(clean_node_text(table.select_one("td.infobox-image .infobox-caption")))
        img = table.select_one("td.infobox-image img")
        image_alt = (img.get("alt") or "").strip() if img and img.has_attr("alt") else ""

        current_section: Optional[str] = None
        grouped: Dict[str, Any] = {}
        flat_rows: List[Dict[str, str]] = []

        for tr in table.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")

            if th and not td:
                sec = self._clean_label(clean_node_text(th))
                if sec and len(sec) <= 120:
                    current_section = sec
                    grouped.setdefault(current_section, {})
                continue

            if not th or not td:
                continue

            label = self._clean_label(clean_node_text(th))
            value = self._clean_value(clean_node_text(td))
            if not label and not value:
                continue

            flat_rows.append({"label": label, "value": value, "section": current_section or ""})

            if current_section:
                sec_map = grouped.setdefault(current_section, {})
                if label in sec_map:
                    if isinstance(sec_map[label], list):
                        sec_map[label].append(value)
                    else:
                        sec_map[label] = [sec_map[label], value]
                else:
                    sec_map[label] = value
            else:
                grouped[label] = value

        return {
            "infobox_title": infobox_title,
            "image_caption": image_caption,
            "image_alt": image_alt,
            "grouped": grouped,
            "rows": flat_rows,
        }

    def _format_infobox_at_top(self, info: Dict[str, Any]) -> str:
        if not info:
            return ""
        lines: List[str] = []
        title = (info.get("infobox_title") or "").strip()
        lines.append(f"[INFOBOX] {title}".strip() if title else "[INFOBOX]")

        cap = (info.get("image_caption") or "").strip()
        if cap:
            lines.append(f"Caption: {cap}")

        grouped = info.get("grouped") or {}

        def emit_kv(k: str, v: Any, indent: int = 0):
            pad = "  " * indent
            if isinstance(v, dict):
                lines.append(f"{pad}{k}:")
                for kk, vv in v.items():
                    emit_kv(kk, vv, indent + 1)
            elif isinstance(v, list):
                lines.append(f"{pad}{k}: " + "; ".join(str(x) for x in v if x))
            else:
                lines.append(f"{pad}{k}: {v}")

        for k, v in grouped.items():
            emit_kv(k, v, 0)

        text = "\n".join(lines).strip()
        if len(text) > self.infobox_preview_max_chars:
            text = text[: self.infobox_preview_max_chars].rstrip() + "..."
        return text

    def _evidence_from_page(self, page: wikipediaapi.WikipediaPage, subject: str, resolved_via: str) -> Evidence:
        url = getattr(page, "fullurl", None) or self._title_to_url(page.title)
        article_text = (page.text or "").strip() or (page.summary or "").strip()

        infobox = {}
        infobox_top = ""
        try:
            infobox = self._extract_infobox_grouped(page.title)
            infobox_top = self._format_infobox_at_top(infobox)
        except Exception as e:
            logger.warning("[Wikipedia._evidence_from_page] infobox failed title=%r error=%r", page.title, e)

        prose_budget = min(self.prose_chars_target, self.max_content_length)
        prose = (article_text or "")[:prose_budget].rstrip()
        if len(article_text or "") > prose_budget:
            prose += "..."

        parts: List[str] = []
        if infobox_top:
            parts.append(infobox_top)
            parts.append("\n\n")
        parts.append(prose)

        content = "".join(parts)
        if len(content) > self.max_content_length:
            content = content[: self.max_content_length].rstrip() + "..."

        rows_list = infobox.get("rows") or []
        rows_dict: Dict[str, Dict[str, str]] = {}
        for i, row in enumerate(rows_list[: self.infobox_rows_preview_limit], start=1):
            rows_dict[str(i)] = {
                "label": str(row.get("label") or ""),
                "value": str(row.get("value") or ""),
                "section": str(row.get("section") or ""),
            }

        extracted: Dict[str, Any] = {
            "tool": self.name,
            "page_title": page.title,
            "subject": subject,
            "resolved_via": resolved_via,
            "url": url,
            "infobox_title": str(infobox.get("infobox_title") or ""),
            "infobox_image_caption": str(infobox.get("image_caption") or ""),
            "infobox_image_alt": str(infobox.get("image_alt") or ""),
            "infobox_grouped": infobox.get("grouped") or {},
            "infobox_rows": rows_dict,
            "entity_hint": page.title,
        }

        return Evidence(
            tool=self.name,
            content=content or f"(Empty content for '{page.title}')",
            url=url,
            extracted=extracted,
            as_of=datetime.utcnow(),
            confidence=0.85,
        )

    def _title_to_url(self, title: str) -> str:
        return self.WIKI_BASE_URL + urllib.parse.quote(title.replace(" ", "_"))

    def _dictify_list_of_strings(self, items: List[str]) -> Dict[str, str]:
        return {str(i + 1): str(v) for i, v in enumerate(items or [])}

    def _evidence_ok(self, *, content: str, url: Optional[str], extracted: Dict[str, Any], confidence: float) -> Evidence:
        return Evidence(
            tool=self.name,
            content=content,
            url=url,
            extracted=extracted or {},
            as_of=datetime.utcnow(),
            confidence=float(confidence),
        )

    def _evidence_error(self, msg: str, reason: str, extra: Optional[dict] = None) -> Evidence:
        ex = {"error": True, "reason": reason, "message": msg}
        if extra:
            ex.update(extra)
        return Evidence(
            tool=self.name,
            content=msg,
            url=None,
            extracted=ex,
            as_of=datetime.utcnow(),
            confidence=0.1,
        )
