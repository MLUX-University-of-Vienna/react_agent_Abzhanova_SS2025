import sys
import logging
from typing import Dict, Tuple, List

from pathlib import Path

from llm_adapters.ollama_depseek_llm import OllamaDeepseekLLM
from llm_adapters.ollama_gemma_llm import OllamaGemmaLLM

from tool_management.tools.calculator_tool import CalculatorTool
from tool_management.tools.serpapi_tool import SerpAPITool
from tool_management.tools.wikipedia_tool import WikipediaTool
from tool_management.tools.vector_search_tool import VectorSearchTool

from agent_memory.pronouns_resolution.spacy_entity_extractor import SpacyEntityExtractor
from agent_memory.pronouns_resolution.pronouns_resolver import PronounsResolver

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
    configure_logging()

    query_separator_llm = OllamaGemmaLLM()
    tool_selector_llm = OllamaGemmaLLM()
    answer_generator_llm = OllamaGemmaLLM()
    confidence_assessor_llm = OllamaGemmaLLM()
    pronouns_resolution_llm = OllamaGemmaLLM()

    tools = create_tools()

    entity_extractor = SpacyEntityExtractor()

    agent = ReActAgent(
        query_separator_llm=query_separator_llm,
        tool_selector_llm=tool_selector_llm,
        answer_generator_llm=answer_generator_llm,
        confidence_assessor_llm=confidence_assessor_llm,
        entity_extractor=entity_extractor,
        pronouns_resolution_llm=pronouns_resolution_llm,
        tools=tools
    )

    print("\n ReAct Agent Chat. Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue  

        response = agent.handle(user_input)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
