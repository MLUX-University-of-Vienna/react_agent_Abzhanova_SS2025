import importlib
import logging
import os
import re
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .llm_adapters.ollama_llama3_llm import OllamaLlama3LLM

from my_react_agent.agent_core.agent_actions import (
    AnswerByItselfAction,
    ClarifyAction,
    UseToolAction,
    StopAction,
)

from my_react_agent.agent_core.agent_actions.need_context_action import NeedContextAction
from my_react_agent.agent_heart.react_agent import ReActAgent, ParameterAssessorFactory

from my_react_agent.agent_memory.llm_entity_extractor import LLMEntityExtractor
from my_react_agent.agent_memory.entity_extractor import EntityExtractor
from my_react_agent.agent_memory.data_structures import Mention, GenderEnum

from my_react_agent.agent_prompts.prompt_registry import PromptRegistry
from my_react_agent.agent_prompts.defaults_prompts import DEFAULT_PROMPTS

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

    for noisy in (
        "urllib3",
        "wikipedia",
        "wikipediaapi",
        "duckduckgo",
        "httpx",
        "httpcore",
        "pdfminer",
        "pdfminer.psparser",
        "pdfminer.pdfdocument",
        "pdfminer.pdfparser",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("[configure_logging] Logging configured level=%s", logging.getLevelName(level))
    logger.info("[configure_logging] Log file: %s", str(log_path))


def _parse_vector_search_files_from_env() -> List[Path]:
    raw = os.getenv("VECTOR_SEARCH_FILES", "").strip()
    if not raw:
        return []
    out: List[Path] = []
    for part in raw.split(","):
        p = part.strip()
        if p:
            out.append(Path(p).expanduser())
    return out


def _init_wikipedia_tool() -> Optional[object]:
    try:
        from .tools.wikipedia_tool import WikipediaTool
        tool = WikipediaTool()
        return tool
    except Exception as e:
        logger.exception("[tools] wikipedia failed to init: %r", e)
        return None


def _init_calculator_tool() -> Optional[object]:
    try:
        from .tools.calculator_tool import CalculatorTool
        tool = CalculatorTool()
        return tool
    except Exception as e:
        logger.exception("[tools] calculator failed to init: %r", e)
        return None


def _init_duckduckgo_tool() -> Optional[object]:
    try:
        from .tools.web_search_tool import DuckDuckGoTool
        tool = DuckDuckGoTool()
        return tool
    except Exception as e:
        logger.exception("[tools] duckduckgo failed to init: %s", e)
        return None


def _init_vector_search_tool() -> Optional[object]:
    try:
        from .tools.vector_search_tool import VectorSearchTool
    except Exception as e:
        logger.exception("[tools] vector_search import failed (VectorSearchTool): %r", e)
        return None

    default_files = [Path(__file__).resolve().parent / "tools" / "facts_about_Austria.pdf"]
    env_files = _parse_vector_search_files_from_env()
    files = env_files if env_files else default_files

    existing = [p for p in files if p.exists()]
    missing = [p for p in files if not p.exists()]
    logger.debug("[tools] vector_search files existing=%d missing=%d", len(existing), len(missing))
    if missing:
        logger.warning("[tools] vector_search missing files: %s", ", ".join(str(p) for p in missing))

    if not existing:
        logger.warning("[tools] vector_search disabled (no existing files)")
        return None

    try:
        tool = VectorSearchTool(existing)
        return tool
    except Exception as e:
        logger.exception("[tools] vector_search failed to init: %r", e)
        return None


def create_tools() -> Dict[str, object]:
    tools: Dict[str, object] = {}

    w = _init_wikipedia_tool()
    if w is not None:
        tools["wikipedia"] = w

    ddg = _init_duckduckgo_tool()
    if ddg is not None:
        tools["web_search"] = ddg

    c = _init_calculator_tool()
    if c is not None:
        tools["calculator"] = c

    vs = _init_vector_search_tool()
    if vs is not None:
        tools["vector_search"] = vs

    logger.info("[create_tools] Tools initialised count=%d keys=%s", len(tools), list(tools.keys()))
    return tools


def create_llms() -> Dict[str, object]:
    planner = OllamaLlama3LLM()
    summariser = OllamaLlama3LLM()
    confidence = OllamaLlama3LLM()
    refiner = OllamaLlama3LLM()

    llms = {
        "planner_llm": planner,
        "summariser_llm": summariser,
        "confidence_llm": confidence,
        "refiner_llm": refiner,
    }

    logger.info(
        "[create_llms] Bound roles planner=%s summariser=%s confidence=%s refiner=%s",
        type(planner).__name__,
        type(summariser).__name__,
        type(confidence).__name__,
        type(refiner).__name__,
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


def create_prompt_registry() -> PromptRegistry:
    defaults = dict(DEFAULT_PROMPTS)
    pr = PromptRegistry(_defaults=defaults)
    logger.info("[create_prompt_registry] loaded default prompts=%d", len(defaults))
    return pr


def create_parameter_assessor_factories() -> List[ParameterAssessorFactory]:
    factories: List[ParameterAssessorFactory] = []

    candidates = [
        ("my_react_agent.confidence_assessment.entity_alignment_assessor", "EntityAlignmentAssessor"),
        ("my_react_agent.confidence_assessment.answer_quality_assessor", "AnswerQualityAssessor"),
        ("my_react_agent.confidence_assessment.answer_realism_assessor", "AnswerRealismAssessor"),
    ]

    for mod_path, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            factories.append(lambda llm, prompts, _cls=cls: _cls(llm, prompts=prompts))
            logger.info("[create_parameter_assessor_factories] enabled %s.%s", mod_path, cls_name)
        except Exception as e:
            logger.warning(
                "[create_parameter_assessor_factories] missing %s.%s (skipping) err=%r",
                mod_path,
                cls_name,
                e,
            )

    logger.info("[create_parameter_assessor_factories] total=%d", len(factories))
    return factories


@dataclass
class PluginContext:
    prompts: PromptRegistry
    tools: Dict[str, object]
    step_actions: List[object]
    low_conf_actions: List[object]
    parameter_assessors: List[ParameterAssessorFactory]


def load_plugins(ctx: PluginContext) -> None:
    raw = os.getenv("REACT_AGENT_PLUGINS", "").strip()
    if not raw:
        logger.info("[plugins] none configured (REACT_AGENT_PLUGINS empty)")
        return

    mods = [x.strip() for x in raw.split(",") if x.strip()]
    logger.info("[plugins] loading count=%d mods=%s", len(mods), mods)

    for name in mods:
        try:
            mod = importlib.import_module(name)
            plugin = getattr(mod, "plugin", None)
            if plugin is None or not hasattr(plugin, "register"):
                logger.warning("[plugins] module=%s has no plugin.register(ctx); skipping", name)
                continue
            plugin.register(ctx)
            logger.info("[plugins] module=%s registered", name)
        except Exception as e:
            logger.exception("[plugins] module=%s failed to load/register err=%r", name, e)


def main() -> None:
    configure_logging(logging.DEBUG)
    logger.info("[main] Starting initialisation sequence")

    prompts = create_prompt_registry()
    tools = create_tools()
    if not tools:
        logger.warning("[main] No tools loaded. Agent can still run with non-tool actions.")

    llms = create_llms()

    try:
        entity_extractor = LLMEntityExtractor(llms["summariser_llm"], prompts=prompts)
        logger.info("[main] LLMEntityExtractor initialised")
    except Exception as e:
        logger.exception("[main] Failed to init LLMEntityExtractor; using RegexEntityExtractor err=%r", e)
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

    parameter_assessors = create_parameter_assessor_factories()

    ctx = PluginContext(
        prompts=prompts,
        tools=tools,
        step_actions=step_actions,
        low_conf_actions=low_conf_actions,
        parameter_assessors=parameter_assessors,
    )
    load_plugins(ctx)

    agent = ReActAgent(
        planner_llm=llms["planner_llm"],
        summariser_llm=llms["summariser_llm"],
        confidence_llm=llms["confidence_llm"],
        refiner_llm=llms["refiner_llm"],
        entity_extractor=entity_extractor,
        tools=tools,
        prompts=prompts,
        max_steps=12,
        step_actions=step_actions,
        low_conf_actions=low_conf_actions,
        parameter_assessors=parameter_assessors,
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
            logger.info("[main] Session terminated by user interrupt")
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