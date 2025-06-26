import re
import logging
from typing import Optional
import wikipedia        
import wikipediaapi

from .agent_tool import AgentTool, ToolResponse


class WikipediaTool(AgentTool):
    def __init__(self, user_agent: str = "AIResearchBot/1.0 (research@example.org)"):
        self.wiki_api = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent=user_agent
        )
        self.logger = logging.getLogger(__name__)
        self.max_summary_length = 500

    @property
    def name(self) -> str:
        return "wikipedia"

    @property
    def description(self) -> str:
        return """Searches Wikipedia for factual information including:
        - Biographies
        - Historical events
        - Scientific concepts
        - Organization information
        - Geographical facts"""

    _encyclopedia_patterns = [
        r'who (is|was|invented|created|discovered|founded)',
        r'what (is|are)( the)? (name|history|origin|meaning|definition|role|title)',
        r'what is the name of',
        r'what does .* mean',
        r'define|explain|describe',
        r'when (was|did)|where (was|did)',
        r'(tell me|what)\s*about',
        r'^(can you|please) (explain|describe)',
        r'\b(biography|invention|discovery|theory|concept)\b'
    ]

    _examples = """User: Who invented Python?
{
  "thought": "This is a historical fact",
  "action": "wikipedia",
  "action_input": "Python (programming language)",
  "final_answer": ""
}"""

    def execute(self, query: str) -> ToolResponse:
        self.logger.info(f"Starting Wikipedia search for query: {query}")
        try:
            search_term = self._prepare_search_term(query)
            if not search_term:
                return ToolResponse(content="No query provided.", success=False)

            results = self._search_wikipedia(search_term)
            if not results:
                return ToolResponse(content=f"No Wikipedia articles found for '{search_term}'", success=True)

            summary = self._get_summary_from_page(results[0])
            return ToolResponse(content=summary, success=True)

        except Exception as e:
            self.logger.error(f"Wikipedia search failed: {e}")
            return ToolResponse(content=f"Error accessing Wikipedia: {e}", success=False)

    def _prepare_search_term(self, query: str) -> str:
        return query.strip()

    def _search_wikipedia(self, term: str) -> list:
        return wikipedia.search(term)

    def _get_summary_from_page(self, page_title: str) -> str:
        page = wikipedia.page(page_title)
        summary = page.summary
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length] + "..."
        return f"Title: {page.title}\nSummary: {summary}"

    @classmethod
    def get_patterns(cls):
        return cls._encyclopedia_patterns

    @classmethod
    def get_examples(cls):
        return cls._examples
