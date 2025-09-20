# main.py
import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Dict, List

from llm_adapters.ollama_mistral_llm import OllamaMistralLLM
from llm_adapters.ollama_deepseek_llm import OllamaDeepseekLLM
from llm_adapters.ollama_llama3_llm import OllamaLlama3LLM

from tool_management.tools.calculator_tool import CalculatorTool
from tool_management.tools.serpapi_tool import SerpAPITool
from tool_management.tools.wikipedia_tool import WikipediaTool
from tool_management.tools.vector_search_tool import VectorSearchTool

from agent_heart.react_agent import ReActAgent

from agent_memory.pronouns_resolution.llm_entity_extractor import LLMEntityExtractor
from agent_memory.pronouns_resolution.entity_extractor import EntityExtractor

from agent_core.agent_actions.answer_by_itself_action import AnswerByItselfAction
from agent_core.agent_actions.clarify_action import ClarifyAction
from agent_core.agent_actions.finalise_action import FinaliseAction
from agent_core.agent_actions.split_query_action import SplitQueryAction
from agent_core.agent_actions.stop_action import StopAction
from agent_core.agent_actions.use_tool_action import UseToolAction

from task_management.logical_form_query_separation import LLMLogicalSeparationStrategy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def configure_logging(level: int = logging.DEBUG) -> None:
    log_path = Path(__file__).resolve().with_name("agent.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(filename=log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.captureWarnings(True)
    for noisy in ("urllib3", "wikipedia", "wikipediaapi", "serpapi", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger.info("[configure_logging] Logging configured and handlers initialised")
    logger.info("[configure_logging] Log file at %s", str(log_path))


def create_tools() -> Dict[str, object]:
    desktop = Path.home() / "Desktop"
    files = [
        desktop / "2025 SS Topics Bachelor Thesis LUX Marian_2.pdf",
        desktop / "firmenbuchpublizitaet_beispiele.docx",
    ]
    serp_api_key = os.getenv("SERPAPI_API_KEY") or "b53c73f7ff8666ff176932332e272be8175325eede05c4f1b716cbbccf6103ca"
    tools = {
        "calculator": CalculatorTool(),
        "wikipedia": WikipediaTool(),
        "web_search": SerpAPITool(api_key=serp_api_key),
        "vector_search": VectorSearchTool(files),
    }
    logger.info(f"[create_tools] Tools initialised count={len(tools)} keys={list(tools.keys())}")
    if not serp_api_key:
        logger.warning("[create_tools] SERPAPI_API_KEY is empty; web_search tool may fail at run-time")
    existing_files = [str(p) for p in files if p.exists()]
    missing_files = [str(p) for p in files if not p.exists()]
    logger.info("[create_tools] Vector search files existing=%d missing=%d", len(existing_files), len(missing_files))
    return tools


def create_llms():
    planner = OllamaLlama3LLM()
    summariser = OllamaLlama3LLM()
    confidence = OllamaLlama3LLM()
    pronouns = OllamaLlama3LLM()
    llms = {
        "planner_llm": planner,
        "summariser_llm": summariser,
        "confidence_llm": confidence,
        "pronouns_llm": pronouns,
    }
    logger.info("[create_llms] LLM roles bound to planner/summariser/confidence/pronouns")
    logger.info("[create_llms] Planner=%s Summariser=%s Confidence=%s Pronouns=%s", type(planner).__name__, type(summariser).__name__, type(confidence).__name__, type(pronouns).__name__)
    return llms


def create_actions(tools: Dict[str, object]):
    tool_keys: List[str] = list(tools.keys())
    actions = [
        UseToolAction(available_tools=tool_keys, tool_registry=tools),
        SplitQueryAction(),
        AnswerByItselfAction(),
        ClarifyAction(),
        FinaliseAction(),
        StopAction(),
    ]
    logger.info(f"[create_actions] Actions initialised count={len(actions)} names={[a.name for a in actions]}")
    return actions


def create_separation_strategy(planner_llm) -> LLMLogicalSeparationStrategy:
    strategy = LLMLogicalSeparationStrategy(llm=planner_llm)
    logger.info("[create_separation_strategy] LLMLogicalSeparationStrategy initialised with %s", type(planner_llm).__name__)
    return strategy


def main():
    configure_logging(logging.DEBUG)
    logger.info("[main] Starting initialisation sequence")
    tools = create_tools()
    llms = create_llms()
    actions = create_actions(tools)
    separation_strategy = create_separation_strategy(llms["planner_llm"])
    try:
        entity_extractor = LLMEntityExtractor(llms["summariser_llm"])
        logger.info("[main] LLMEntityExtractor initialised")
    except TypeError:
        entity_extractor = EntityExtractor(llm=llms["summariser_llm"])
        logger.info("[main] Fallback EntityExtractor initialised")
    agent = ReActAgent(
        planner_llm=llms["planner_llm"],
        summariser_llm=llms["summariser_llm"],
        confidence_llm=llms["confidence_llm"],
        pronouns_llm=llms["pronouns_llm"],
        entity_extractor=entity_extractor,
        tools=tools,
        separation_strategy=separation_strategy,
        actions=actions,
        max_steps=12,
        confidence_threshold=0.80,
    )
    logger.info("[main] ReActPaperAgent constructed and ready")
    print("\n ReAct Paper-Style Agent Chat (Think–Act–Observe). Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            logger.info("[main] Session terminated by user via interrupt")
            break
        if not user_input:
            logger.info("[main] Empty input ignored")
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            logger.info("[main] User requested exit")
            break
        logger.info(f"[main] Handling user query length={len(user_input)}")
        reply = agent.handle(user_input)
        print(f"{reply}\n")


if __name__ == "__main__":
    main()
