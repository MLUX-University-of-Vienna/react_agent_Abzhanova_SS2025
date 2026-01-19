import logging
import os
import sys
import re
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, List

from llm_adapters.ollama_mistral_llm import OllamaMistralLLM
from llm_adapters.ollama_deepseek_llm import OllamaDeepseekLLM
from llm_adapters.ollama_llama3_llm import OllamaLlama3LLM

from agent_core.agent_actions import (
    Action,
    AnswerByItselfAction,
    ClarifyAction,
    UseToolAction,
    StopAction
)

from agent_core.agent_actions.need_context_action import NeedContextAction
from agent_heart.react_agent import ReActAgent

from agent_memory.llm_entity_extractor import LLMEntityExtractor
from agent_memory.entity_extractor import EntityExtractor
from agent_memory.data_structures import Mention
from agent_memory.data_structures import GenderEnum

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def configure_logging(level: int = logging.DEBUG) -> None:
    log_path = Path(__file__).resolve().with_name("agent.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            filename=log_path,
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        ),
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


def _parse_vector_search_files_from_env() -> List[Path]:
    raw = os.getenv("VECTOR_SEARCH_FILES", "").strip()
    if not raw:
        return []
    out: List[Path] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(Path(p).expanduser())
    return out


def create_tools() -> Dict[str, object]:
    tools: Dict[str, object] = {}

    serp_api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    files = _parse_vector_search_files_from_env()

    try:
        from tool_management.tools.calculator_tool import CalculatorTool
        from tool_management.tools.serpapi_tool import SerpAPITool
        from tool_management.tools.wikipedia_tool import WikipediaTool
        from tool_management.tools.vector_search_tool import VectorSearchTool

        tools = {
            "wikipedia": WikipediaTool(),
            "web_search": SerpAPITool(api_key=serp_api_key),
            "vector_search": VectorSearchTool(files),
            "calculator": CalculatorTool(),
        }

    except Exception as e:
        logger.exception("[create_tools] Failed to import/init tools; continuing with no tools. err=%s", e)
        tools = {}

    logger.info(
        "[create_tools] Tools initialised count=%d keys=%s",
        len(tools),
        list(tools.keys()),
    )

    if not serp_api_key:
        logger.warning(
            "[create_tools] SERPAPI_API_KEY is empty; web_search tool may fail at run-time"
        )

    existing_files = [str(p) for p in files if p.exists()]
    missing_files = [str(p) for p in files if not p.exists()]
    logger.info(
        "[create_tools] Vector search files existing=%d missing=%d",
        len(existing_files),
        len(missing_files),
    )
    if missing_files:
        logger.info(
            "[create_tools] Missing vector search files: %s",
            ", ".join(missing_files),
        )

    return tools


def create_llms():
    planner = OllamaLlama3LLM()
    summariser = OllamaLlama3LLM()
    confidence = OllamaLlama3LLM()
    entity_extractor = OllamaLlama3LLM()

    llms = {
        "planner_llm": planner,
        "summariser_llm": summariser,
        "confidence_llm": confidence,
        "entity_extraction_llm": entity_extractor,
    }

    logger.info("[create_llms] LLM roles bound to planner/summariser/confidence/entity_extractor")
    logger.info(
        "[create_llms] Planner=%s Summariser=%s Confidence=%s Entity_Extraction_Llm=%s",
        type(planner).__name__,
        type(summariser).__name__,
        type(confidence).__name__,
        type(entity_extractor).__name__,
    )

    return llms


class RegexEntityExtractor(EntityExtractor):

    _proper = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")

    def extract(self, text: str) -> Optional[Mention]:
        if not text:
            return None
        m = self._proper.search(text)
        if not m:
            return None
        span = m.group(1)
        return Mention(
            text=span,
            start=m.start(1),
            end=m.end(1),
            gender=GenderEnum.UNKNOWN,
            entity_id=None,
        )


def main():
    configure_logging(logging.DEBUG)
    logger.info("[main] Starting initialisation sequence")

    tools = create_tools()
    llms = create_llms()

    try:
        entity_extractor = LLMEntityExtractor(llms["summariser_llm"])
        logger.info("[main] LLMEntityExtractor initialised")
    except Exception as e:
        logger.exception("[main] Failed to init LLMEntityExtractor; using RegexEntityExtractor. err=%s", e)
        entity_extractor = RegexEntityExtractor()
        logger.info("[main] RegexEntityExtractor initialised")

    step_actions = [
        NeedContextAction(),
        AnswerByItselfAction(),
        ClarifyAction(),
        UseToolAction(),
        StopAction(),
    ]

    low_conf_actions = [
        NeedContextAction(),
        UseToolAction(),
        AnswerByItselfAction(),
        StopAction(),
        ClarifyAction(),
    ]

    agent = ReActAgent(
        planner_llm=llms["planner_llm"],
        summariser_llm=llms["summariser_llm"],
        confidence_llm=llms["confidence_llm"],
        entity_extractor=entity_extractor,
        tools=tools,
        max_steps=12,
        step_actions=step_actions,
        low_conf_actions=low_conf_actions,
    )

    logger.info("[main] ReActAgent constructed and ready")

    print(
        "\nReAct Plan–Execute Agent Chat.\n"
        "Flow: plan steps → execute each step → finalise.\n"
        "Type 'exit' or 'quit' to stop.\n"
    )

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

        logger.info("[main] Handling user query length=%d", len(user_input))
        reply = agent.handle(user_input)
        print(f"{reply}\n")


if __name__ == "__main__":
    main()
