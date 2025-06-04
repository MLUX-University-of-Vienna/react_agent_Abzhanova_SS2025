import logging
from typing import List, Dict, Optional
from serpapi.google_search import GoogleSearch  # Correct import
from tool_management.tools.agent_tool import AgentTool, ToolResponse

class SerpAPITool(AgentTool):
    def __init__(self, api_key: str, max_results: int = 3):
        self.api_key = api_key
        self.max_results = max_results
        self.logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "google_search"

    @property
    def description(self) -> str:
        return ("Performs live web searches for current events, breaking news, "
                "real-time data and recent information.")

    @classmethod
    def get_patterns(cls) -> List[str]:
        return [
            r'\b(current|latest|today\'?s|recent|live)\b.*\b(news|events|weather|stock price|headlines|developments|updates|scores?)\b',
            r'\b(news|weather|stocks?|headlines|score|trend|update|price|forecast)\b.*\b(now|today|currently|live|latest|this week|this moment)\b',

            r'\b(what is|what\'s|how is|how\'s|give me|show me|tell me|can you)\b.*\b(news|weather|stock|score|price|trending)\b.*(now|today|currently|live)?',

            r'\b(live|current|today\'?s)\b.*\b(game|match|score|tournament|results?)\b',

            r'\b(price|value|rate|exchange rate)\b.*\b(of|for)\b.*\b(stock|bitcoin|ethereum|usd|eur|market)\b',
        ]


    @classmethod
    def get_examples(cls) -> str:
        return (
            "User: Current weather in Paris?\n"
            "{\n"
            '  "thought": "This requires live data",\n'
            '  "action": "web_search",\n'
            '  "action_input": "current weather Paris",\n'
            '  "final_answer": ""\n'
            "}"
        )

    def _format_results(self, results: List[Dict]) -> str:
        if not results:
            return "No results found."
            
        formatted = []
        for item in results[:self.max_results]:
            title = item.get("title", "No title")
            snippet = item.get("snippet", "No details")
            link = item.get("link", "#")
            formatted.append(f"• {title}\n  {snippet}\n  Source: {link}")
        return "\n\n".join(formatted)

    def execute(self, query: str) -> ToolResponse:
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "num": self.max_results
            }
            
            client = GoogleSearch(params)
            results = client.get_dict().get("organic_results", [])
            
            return ToolResponse(
                content=self._format_results(results),
                success=True,
                metadata={"raw_results": results}
            )
            
        except Exception as e:
            return ToolResponse(
                content=f"Search failed: {str(e)}",
                success=False
            )