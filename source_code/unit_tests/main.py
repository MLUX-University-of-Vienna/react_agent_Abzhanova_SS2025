import sys
import logging
from typing import Dict, Tuple, List

from llm.ollama.ollama_interface import OllamaInterface
from tool_management.tools.calculator_tool import CalculatorTool
from tool_management.tools.serpapi_tool import SerpAPITool
from tool_management.tools.wikipedia_tool import WikipediaTool
from agent_heart.react_agent import ReActAgent

def configure_logging():
    """Configure logging for the entire application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def create_tools() -> Dict[str, object]:
    """Instantiate and return all available AgentTool implementations."""
    return {
        "calculator": CalculatorTool(),
        "wikipedia": WikipediaTool(),
        "web_search": SerpAPITool(
            api_key="b53c73f7ff8666ff176932332e272be8175325eede05c4f1b716cbbccf6103ca"
        )
    }

def run_test_case(agent: ReActAgent, query: str, expected_tool: str):
    """Run a query through the agent and print out the result."""
    print("\n" + "=" * 80)
    print(f"🔍 Testing: {query!r}  (expected tool: {expected_tool})\n")
    answer = agent.handle(query)
    print(f"🤖 Final Answer:\n{answer}\n")
    print("=" * 80 + "\n")

def main():
    configure_logging()

    # 1. Initialize the LLM
    llm = OllamaInterface(model="deepseek-r1:8b", temperature=0.1)

    # 2. Instantiate tools
    tools = create_tools()

    # 3. Build the ReActAgent
    agent = ReActAgent(llm, tools)

    # 4. Define & run your test cases
    test_cases: List[Tuple[str, str]] = [
        ("Who is the president of the USA?", "web_search"),
        # …add more as needed
    ]

    for query, expected in test_cases:
        run_test_case(agent, query, expected)

if __name__ == "__main__":
    main()
