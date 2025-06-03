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

    def execute(self, query: str) -> ToolResponse:
        try:
            self.logger.info(f"Starting Wikipedia search for query: {query}")

            page = self.wiki_api.page(term)
            if not page.exists():
                self.logger.info(f"Page not found, searching via wikipedia.search() for: {term}")
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
