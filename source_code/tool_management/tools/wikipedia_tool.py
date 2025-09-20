import logging
import re
import urllib.parse
from datetime import datetime
from typing import List, Optional, Tuple

import wikipedia
import wikipediaapi

from .agent_tool import AgentTool
from agent_core.models import ToolResponse, Observation

logger = logging.getLogger(__name__)


class WikipediaTool(AgentTool):
    """
    Input (from refiner):
      - "title: <Exact Title>"
      - "<Exact Title>"
      - Multiple candidates: "title: A | B | C"

    Goals:
      - Be robust to query-like inputs (e.g., "population of Vienna", "capital of Austria"):
        * Do NOT hardcode any entity knowledge.
        * Heuristically extract a plausible entity candidate (e.g., "Vienna", "Austria") and let the rest of the system use it.
      - Reject placeholders/templates or joined prompts (e.g., "[city]", "A + B").
      - Resolve via exact → suggest() → search(), skipping disambiguation pages.
      - Return concise summary + URL with clear `metadata.reason` and `metadata.resolved_via`.
    """

    # ---------- config ----------
    def __init__(self, user_agent: str = "AIResearchBot/1.0 (research@example.org)"):
        self.wiki_api = wikipediaapi.Wikipedia(
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent=user_agent,
        )
        self.max_summary_length = 600
        self.search_results = 10

    # ---------- tool contract ----------
    @property
    def name(self) -> str:
        return "wikipedia"

    @property
    def description(self) -> str:
        return "Fetches a Wikipedia page by exact title and returns a concise summary with URL."

    @property
    def refiner_instructions(self) -> str:
        return "Output 'title: <Exact Wikipedia page title>' (optionally multiple with '|'). No questions or prose."

    @property
    def refiner_input_format(self) -> str:
        return "title: <Exact Page Title>  |  <Exact Page Title>"

    @property
    def refiner_input_regex(self) -> Optional[str]:
        # Optional 'title:' then up to 3 titles separated by '|'. Forbid line breaks / braces / angle brackets.
        return r"^(?:title:\s*)?[^\n\r{}<>]{1,200}(?:\s*\|\s*[^\n\r{}<>]{1,200}){0,2}$"

    @property
    def refiner_forbidden(self) -> str:
        return "questions, multi_line, json, placeholders, plus_sign, wikipedia_prefix"

    @property
    def refiner_examples(self) -> str:
        return "\n".join([
            "title: Vienna",
            "United States",
            "title: Python (programming language) | Guido van Rossum",
        ])

    # ---------- execution ----------
    def execute(self, tool_input: str) -> ToolResponse:
        logger.info(f"[Wikipedia] input={tool_input!r}")
        raw = (tool_input or "").strip()
        if not raw:
            return self._fail("No title provided.", reason="empty_input")

        candidates = self._parse_titles_or_coerce(raw)
        if not candidates:
            return self._fail(
                "Invalid Wikipedia title input. Provide an exact title (optionally multiple with '|').",
                reason="invalid_format",
                extra={"input": raw},
            )

        # Try to resolve each in order
        for candidate in candidates:
            page, how = self._resolve_page(candidate)
            if page and page.exists() and not self._is_disambiguation(page):
                return self._success(page, subject=candidate, resolved_via=how)

        return self._ok_with_message(
            f"No page found for any of: {', '.join(map(repr, candidates))}. "
            f"Provide an exact title such as 'Vienna' or 'United States'.",
            reason="not_found",
            extra={"candidates": candidates},
        )

    # ---------- helpers ----------
    _TITLE_SPLIT_RX = re.compile(r"\s*\|\s*")
    _PLACEHOLDER_RX = re.compile(r"\[[^\]]+\]|\{[^}]+\}", re.I)
    _JOINED_PROMPT_RX = re.compile(r"\s(?:\+| and |,)\s", re.I)
    _STRIP_PARENS_RX = re.compile(r"\((.*?)\)")
    _MAY_REFER_TO_RX = re.compile(r"\bmay refer to:\b", re.I)

    # Generic phrase extractor: keep the last NP-ish tail after common prepositions.
    # Examples:
    #   "population of Vienna"        -> "Vienna"
    #   "capital city of the Austria" -> "the Austria" (later normalized to "Austria")
    #   "weather in New York City"    -> "New York City"
    _TAIL_OF_PREP_RX = re.compile(r".*\b(?:of|in|for|about|on|at|from|near)\s+(.+)$", re.I)

    def _parse_titles_or_coerce(self, q: str) -> List[str]:
        """
        Accept:
          - 'title: Vienna'
          - 'Vienna'
          - 'title: Vienna | United States'

        Also tolerate query-like inputs by extracting a plausible entity (no hardcoded knowledge).
        Reject obvious placeholders or joined prompts.
        """
        # Allow optional 'title:' prefix
        m = re.match(r"^title:\s*(.+)$", q, flags=re.I)
        s = m.group(1).strip() if m else q

        # Hard reject placeholders and joined prompts like 'A + B' or 'A and B'
        if self._PLACEHOLDER_RX.search(s) or self._JOINED_PROMPT_RX.search(s):
            return []

        # If it already looks like multiple candidates separated by '|'
        if "|" in s:
            parts = [p.strip() for p in self._TITLE_SPLIT_RX.split(s) if p.strip()]
            return parts[:3]

        # Otherwise, try to coerce a query-like phrase into a single candidate title:
        candidate, _why = self._coerce_candidate_to_title(s)
        return [candidate] if candidate else []

    def _coerce_candidate_to_title(self, text: str) -> Tuple[Optional[str], str]:
        """
        Generic, language-agnostic heuristics. No entity hardcoding.
        - Strip 'Wikipedia:' prefix.
        - Drop parenthetical hints.
        - If there's a prepositional phrase, keep the tail after the last preposition.
        - Remove leading articles and generic modifiers like 'the', 'current', 'total' if they are at the very start.
        """
        s = text.strip()

        # Remove accidental "Wikipedia:" namespace prefix
        if s.lower().startswith("wikipedia:"):
            s = s.split(":", 1)[1].strip()

        # Strip simple parentheticals like "(capital city of Austria)"
        s = self._STRIP_PARENS_RX.sub("", s).strip()

        # If query-like, keep the tail after the last common preposition
        m = self._TAIL_OF_PREP_RX.match(s)
        if m:
            s = m.group(1).strip()

        # Trim very generic leading tokens (no hardcoding of specific entities)
        s = re.sub(r"^(?:the\s+)?(?:current|latest|total)\s+", "", s, flags=re.I).strip()

        # If the remaining looks like a title-ish string (not empty, no pipes)
        if s and "|" not in s:
            return s, "coerced_from_query"
        return None, "unusable_after_coercion"

    def _resolve_page(self, title: str) -> Tuple[Optional[wikipediaapi.WikipediaPage], str]:
        """Exact → suggest() → search(); skip disambiguation pages."""
        # Exact
        page = self.wiki_api.page(title)
        if page and page.exists() and not self._is_disambiguation(page):
            return page, "exact"

        # Suggest
        try:
            suggestion = wikipedia.suggest(title)
        except Exception:
            suggestion = None
        if suggestion:
            pg = self.wiki_api.page(suggestion)
            if pg and pg.exists() and not self._is_disambiguation(pg):
                return pg, "suggest"

        # Search (top-N)
        try:
            results = wikipedia.search(title, results=self.search_results)
        except Exception:
            results = []
        for r in results or []:
            if not r:
                continue
            if "disambiguation" in r.lower():
                continue
            pg = self.wiki_api.page(r)
            if pg and pg.exists() and not self._is_disambiguation(pg):
                return pg, "search"

        return None, "unresolved"

    def _is_disambiguation(self, page: wikipediaapi.WikipediaPage) -> bool:
        try:
            cats = getattr(page, "categories", {})
            if any("disambiguation" in k.lower() for k in (cats or {}).keys()):
                return True
        except Exception:
            pass
        try:
            txt = (page.summary or page.text or "")[:600].lower()
            if self._MAY_REFER_TO_RX.search(txt):
                return True
        except Exception:
            pass
        return False

    def _success(self, page: wikipediaapi.WikipediaPage, subject: str, resolved_via: str) -> ToolResponse:
        url = getattr(page, "fullurl", None) or self._title_to_url(page.title)
        summary = (page.summary or "").strip()
        if len(summary) > self.max_summary_length:
            summary = summary[: self.max_summary_length].rstrip() + "..."
        obs = Observation(
            raw_text=summary or f"(Empty summary for '{page.title}')",
            title=page.title,
            source="wikipedia",
            source_url=url,
            observed_at=datetime.utcnow(),
        )
        return ToolResponse(
            success=True,
            observations=[obs],
            metadata={
                "tool": self.name,
                "subject": subject,
                "page_title": page.title,
                "source_url": url,
                "resolved_via": resolved_via,
            },
        )

    def _ok_with_message(self, msg: str, reason: str, extra: Optional[dict] = None) -> ToolResponse:
        obs = Observation(raw_text=msg, source="wikipedia", observed_at=datetime.utcnow())
        md = {"tool": self.name, "reason": reason}
        if extra:
            md.update(extra)
        return ToolResponse(success=True, observations=[obs], metadata=md)

    def _fail(self, msg: str, reason: str, extra: Optional[dict] = None) -> ToolResponse:
        obs = Observation(raw_text=msg, source="wikipedia", observed_at=datetime.utcnow())
        md = {"tool": self.name, "reason": reason}
        if extra:
            md.update(extra)
        return ToolResponse(success=False, observations=[obs], metadata=md)

    def _title_to_url(self, title: str) -> str:
        return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
