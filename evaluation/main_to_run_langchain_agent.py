from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

try:
    from langchain.agents import initialize_agent
except Exception as e:
    raise RuntimeError(
        "Could not import initialize_agent from langchain.agents. "
        "Please install/upgrade langchain."
    ) from e

try:
    from langchain.tools import Tool
except Exception:
    from langchain_core.tools import Tool 

try:
    from langchain_ollama import ChatOllama
except Exception:
    from langchain_community.chat_models import ChatOllama  

from evaluation.tools.calculator_tool import CalculatorTool
from evaluation.tools.vector_search_tool import VectorSearchTool
from evaluation.tools.web_search_tool import DuckDuckGoTool
from evaluation.tools.wikipedia_tool import WikipediaTool

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    log_path = Path(__file__).resolve().with_name("langchain_agent.log")
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
        "httpx",
        "httpcore",
        "pdfminer",
        "wikipedia",
        "wikipediaapi",
        "langchain",
        "langsmith",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("[configure_logging] level=%s log_file=%s", logging.getLevelName(level), log_path)


def create_my_tools() -> Dict[str, object]:
    tools: Dict[str, object] = {}

    try:
        tools["wikipedia"] = WikipediaTool()
    except Exception:
        logger.exception("[tools] wikipedia init failed")

    try:
        tools["web_search"] = DuckDuckGoTool()
    except Exception:
        logger.exception("[tools] web_search init failed")

    try:
        tools["calculator"] = CalculatorTool()
    except Exception:
        logger.exception("[tools] calculator init failed")

    # vector search default: bundled Austria PDF
    try:
        default_pdf = Path(__file__).resolve().parent / "tools" / "facts_about_Austria.pdf"
        if default_pdf.exists():
            tools["vector_search"] = VectorSearchTool([default_pdf])
        else:
            logger.warning("[tools] vector_search disabled (missing file): %s", default_pdf)
    except Exception:
        logger.exception("[tools] vector_search init failed")

    logger.info("[create_my_tools] count=%d keys=%s", len(tools), list(tools.keys()))
    return tools


# input normalisation
def _to_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("input", "query", "text", "question"):
            if k in x and x[k] is not None:
                return str(x[k])
        return str(x)
    return str(x)

def wrap_tools_for_langchain(
    my_tools: Dict[str, object],
) -> List[Tool]:
    """
    1) TOOL ALIASES (case + common names):
        register both "Wikipedia" and "wikipedia" (same function).

    2) NORMALISED INPUT:
       Prevents errors when LangChain sends dict/object inputs.
    """

    wiki_tool_obj = my_tools.get("wikipedia")
    web_tool_obj = my_tools.get("web_search")
    calc_tool_obj = my_tools.get("calculator")
    vec_tool_obj = my_tools.get("vector_search")

    lc_tools: List[Tool] = []

    def make_execute_fn(tool_obj: object) -> Callable[[Any], str]:
        def _fn(tool_input: Any) -> str:
            text = _to_text(tool_input).strip()

            ev = tool_obj.execute(text)
            content = getattr(ev, "content", None)
            if content is None:
                content = str(ev)
            else:
                content = str(content)

            url = getattr(ev, "url", None)
            if isinstance(url, str) and url.strip():
                content = f"{content}\n\nSOURCE: {url.strip()}"

            return content

        return _fn

    # Wikipedia (aliases)
    if wiki_tool_obj is not None:
        wiki_fn = make_execute_fn(wiki_tool_obj)
        lc_tools.append(Tool(name="Wikipedia", func=wiki_fn, description="Look up facts on Wikipedia."))
        lc_tools.append(Tool(name="wikipedia", func=wiki_fn, description="(alias) Look up facts on Wikipedia."))

    # Calculator (aliases)
    if calc_tool_obj is not None:
        calc_fn = make_execute_fn(calc_tool_obj)
        lc_tools.append(Tool(name="Calculator", func=calc_fn, description="Do math calculations."))
        lc_tools.append(Tool(name="calculator", func=calc_fn, description="(alias) Do math calculations."))

    # Vector search (aliases) 
    if vec_tool_obj is not None:
        vec_fn = make_execute_fn(vec_tool_obj)
        lc_tools.append(Tool(name="VectorSearch", func=vec_fn, description="Search your local documents."))
        lc_tools.append(Tool(name="vector_search", func=vec_fn, description="(alias) Search your local documents."))

    logger.info("[wrap_tools_for_langchain] count=%d", len(lc_tools))
    return lc_tools


def create_ollama_llm() -> ChatOllama:
    model = "llama3:8b"
    base_url = "http://localhost:11434"
    timeout = 120 

    # Avoid LangSmith noise without requiring env setup
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

    llm = ChatOllama(model=model, base_url=base_url, timeout=timeout)
    logger.info("[llm] ChatOllama model=%s base_url=%s timeout=%s", model, base_url, timeout)
    return llm


async def main_async() -> None:
    configure_logging(logging.DEBUG)

    logger.info("[main] init tools")
    my_tools = create_my_tools()
    tools = wrap_tools_for_langchain(my_tools)

    llm = create_ollama_llm()

    # It uses LangChain's built-in prompt for
    # "zero-shot-react-description" (MRKL ReAct style).
    # handle_parsing_errors:
    # When the model outputs text that doesn't follow:
    #   Thought: ...
    #   Action: ...
    #   Action Input: ...
    executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        max_iterations=20,
        handle_parsing_errors=(
            "Your previous output was not in the required ReAct format. "
            "You MUST respond using exactly: Thought, Action, Action Input. "
            "Action MUST be one of the provided tool names."
        ),
        early_stopping_method="force", 
    )

    print(
        "\nLangChain ReAct Agent Chat (initialize_agent + zero-shot-react-description, no custom prompt).\n"
        "Type 'exit' or 'quit' to stop.\n"
        f"Model: {getattr(llm, 'model', 'unknown')}\n"
        f"Max iterations: {getattr(executor, 'max_iterations', 'unknown')} | Verbose: {getattr(executor, 'verbose', False)}\n"
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

        try:
            logger.info("[chat] user_len=%d", len(user_input))
            result = executor.invoke({"input": user_input})
            print(f"{result.get('output', '')}\n")
        except Exception as e:
            logger.exception("[chat] failed")
            print(f"[ERROR] {e}\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
