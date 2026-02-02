from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

from my_react_agent.agent_heart.react_agent import ReActAgent
from my_react_agent.tool_management.tools.agent_tool import AgentTool


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

    logger.info("[configure_logging] Log file: %s", str(log_path))


def _parse_vector_search_files_from_env() -> List[Path]:
    raw = os.getenv("VECTOR_SEARCH_FILES", "").strip()
    if not raw:
        return []
    return [Path(p.strip()).expanduser() for p in raw.split(",") if p.strip()]


def _try_init(tool_cls, *, name: str):
    try:
        t = tool_cls()
        logger.info("[tools] enabled %s", name)
        return t
    except Exception as e:
        logger.exception("[tools] %s failed to init: %r", name, e)
        return None


def build_tools() -> List[AgentTool]:
    tools: List[AgentTool] = []

    # Wikipedia
    try:
        from evaluation.tools.wikipedia_tool import WikipediaTool
        t = _try_init(WikipediaTool, name="wikipedia")
        if t:
            tools.append(t)
    except Exception:
        logger.exception("[tools] wikipedia import failed")

    # Web search
    try:
        from evaluation.tools.web_search_tool import DuckDuckGoTool
        t = _try_init(DuckDuckGoTool, name="web_search")
        if t:
            tools.append(t)
    except Exception:
        logger.exception("[tools] web_search import failed")

    # Calculator
    try:
        from evaluation.tools.calculator_tool import CalculatorTool
        t = _try_init(CalculatorTool, name="calculator")
        if t:
            tools.append(t)
    except Exception:
        logger.exception("[tools] calculator import failed")

    # Vector search 
    try:
        from evaluation.tools.vector_search_tool import VectorSearchTool
        default_files = [Path(__file__).resolve().parent / "tools" / "facts_about_Austria.pdf"]
        files = _parse_vector_search_files_from_env() or default_files
        existing = [p for p in files if p.exists()]

        if not existing:
            logger.warning("[tools] vector_search disabled (no existing files)")
        else:
            tools.append(VectorSearchTool(existing))
            logger.info("[tools] enabled vector_search files=%d", len(existing))
    except Exception as e:
        logger.exception("[tools] vector_search failed to init: %r", e)

    if not tools:
        raise RuntimeError("No tools loaded. ReActAgent requires at least one tool.")

    logger.info("[tools] total=%d names=%s", len(tools), [t.name for t in tools])
    return tools


def main() -> None:
    configure_logging(logging.DEBUG)

    model = os.getenv("REACT_MODEL", "llama3:8b")
    tools = build_tools()

    agent = ReActAgent.create(
        llm=model,
        tools=tools,
    )

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
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        reply = agent.handle(user_input)
        print(f"{reply}\n")


if __name__ == "__main__":
    main()
