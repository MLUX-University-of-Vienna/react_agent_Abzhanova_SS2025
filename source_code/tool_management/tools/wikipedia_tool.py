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
        r'what (is|are) (the )?(history|origin|meaning|definition)',
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

    def _extract_search_term(self, query: str) -> str:
        q = re.sub(r'^(who|what|when|where|why|how)\s+', '', query, flags=re.IGNORECASE)
        q = re.sub(r'[?.,;!]', '', q).strip()
        for pattern, repl in [
            (r'invented (.*)', r'\1'),
            (r'explain (.*)', r'\1'),
            (r'current (.*) of (.*)', r'\2 \1'),
            (r'capital of (.*)', r'\1')
        ]:
            if re.search(pattern, q, re.IGNORECASE):
                self.logger.debug(f"Reformatted search term using pattern: {pattern}")
                return re.sub(pattern, repl, q, flags=re.IGNORECASE).title()
        self.logger.debug("Returning unmodified search term")
        return q.title()

    def execute(self, query: str) -> ToolResponse:
        try:
            self.logger.info(f"Starting Wikipedia search for query: {query}")
            term = self._extract_search_term(query)
            self.logger.info(f"Processed Wikipedia search term: {term}")

            # Try direct page lookup first
            page = self.wiki_api.page(term)
            if not page.exists():
                self.logger.info(f"Page not found, searching via wikipedia.search() for: {term}")
                # fallback to wikipedia lib
                results = wikipedia.search(term)
                if not results:
                    self.logger.warning(f"No search results for '{term}'")
                    return ToolResponse(content=f"No Wikipedia info found for '{term}'", success=False)
                term = results[0]
                self.logger.info(f"Using top search result: {term}")
                page = self.wiki_api.page(term)

            if page.exists():
                summary = page.summary or page.text.split('\n')[0]
                summary = re.sub(r'\[\d+\]', '', summary)[:self.max_summary_length].strip()
                url = page.fullurl
                self.logger.info(f"Fetched summary for: {term}")
                return ToolResponse(content=f"{summary}\n\nSource: {url}", success=True)

            self.logger.warning(f"No page exists for '{term}' even after search")
            return ToolResponse(content=f"No Wikipedia info found for '{term}'", success=False)

        except Exception as e:
            self.logger.error(f"Wikipedia search failed: {e}")
            return ToolResponse(content=f"Error accessing Wikipedia: {e}", success=False)

    @classmethod
    def get_patterns(cls):
        return cls._encyclopedia_patterns

    @classmethod
    def get_examples(cls):
        return cls._examples
