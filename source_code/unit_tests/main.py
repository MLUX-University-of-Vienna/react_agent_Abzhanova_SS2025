import sys
import logging
from typing import Dict, Tuple, List

from pathlib import Path

from llm.ollama.ollama_interface import OllamaInterface
from llm.gemma_interface import GemmaInterface

from tool_management.tools.calculator_tool import CalculatorTool
from tool_management.tools.serpapi_tool import SerpAPITool
from tool_management.tools.wikipedia_tool import WikipediaTool
from tool_management.tools.vector_search_tool import VectorSearchTool

from agent_heart.react_agent import ReActAgent

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def create_tools() -> Dict[str, object]:
    desktop = Path.home() / "Desktop"

    files = [
        desktop / "2025 SS Topics Bachelor Thesis LUX Marian_2.pdf",
        desktop / "firmenbuchpublizitaet_beispiele.docx"
    ]

    return {
        "calculator": CalculatorTool(),
        "wikipedia": WikipediaTool(),
        "web_search": SerpAPITool(
            api_key="b53c73f7ff8666ff176932332e272be8175325eede05c4f1b716cbbccf6103ca"
        ),
        "vector_search": VectorSearchTool(files)
    }


def run_test_case(agent: ReActAgent, query: str, expected_tool: str):
    """Run a query through the agent and print out the result"""
    print("\n" + "=" * 80)
    print(f" Testing: {query!r}  (expected tool: {expected_tool})\n")
    answer = agent.handle(query)
    print(f"Final Answer:\n{answer}\n")
    print("=" * 80 + "\n")

def main():

    query_separator_llm = GemmaInterface(model="gemma3:4b", temperature=0.1)
    tool_selector_llm = GemmaInterface(model="gemma3:4b", temperature=0.1)
    answer_generator_llm = GemmaInterface(model="gemma3:4b", temperature=0.1)
    confidence_assessor_llm = GemmaInterface(model="gemma3:4b", temperature=0.1)

    tools = create_tools()

    agent = ReActAgent(
        query_separator_llm=query_separator_llm,
        tool_selector_llm=tool_selector_llm,
        answer_generator_llm=answer_generator_llm,
        confidence_assessor_llm=confidence_assessor_llm,
        tools=tools
    )

    test_cases: List[Tuple[str, str]] = [
        ("Who is the president of USA and how old is he?", "web_search"),
    ]

    for query, expected in test_cases:
        run_test_case(agent, query, expected)

if __name__ == "__main__":
    main()
