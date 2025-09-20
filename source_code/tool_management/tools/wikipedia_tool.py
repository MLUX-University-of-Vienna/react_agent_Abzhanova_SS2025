import re
import logging
from typing import List
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
        return (
            "Searches Wikipedia for factual information including:\n"
            "- Biographies\n"
            "- Historical events\n"
            "- Scientific concepts\n"
            "- Organization information\n"
            "- Geographical facts"
        )

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
        self.logger.info(f"Starting Wikipedia search for query: '{query}'")
        term = query.strip()
        if not term:
            return ToolResponse(content="No query provided.", success=False)

        try:
            titles: List[str] = wikipedia.search(term, results=5)
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return ToolResponse(content=f"Error during search: {e}", success=False)

        if not titles:
            return ToolResponse(content=f"No Wikipedia articles found for '{term}'", success=True)

        page = self.wiki_api.page(titles[0])
        if not page.exists():
            return ToolResponse(content=f"Found title '{titles[0]}' but page does not exist.", success=False)

        summary = page.summary
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length].rstrip() + "..."

        content = f"Title: {page.title}\nSummary: {summary}"
        return ToolResponse(content=content, success=True)

    @classmethod
    def get_patterns(cls):
        return cls._encyclopedia_patterns

    @classmethod
    def get_examples(cls):
        return cls._examples
