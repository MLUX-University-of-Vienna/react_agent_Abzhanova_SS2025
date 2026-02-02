from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Callable

from evaluation.tools.calculator_tool import CalculatorTool
from evaluation.tools.vector_search_tool import VectorSearchTool
from evaluation.tools.web_search_tool import DuckDuckGoTool
from evaluation.tools.wikipedia_tool import WikipediaTool
from langchain.agents import create_agent
from langchain_core.tools import Tool

try:
    from langchain_ollama import ChatOllama
except Exception:
    from langchain_community.chat_models import ChatOllama  


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

    logger.info(
        "[configure_logging] level=%s log_file=%s",
        logging.getLevelName(level),
        log_path,
    )


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


def _to_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("input", "query", "text", "question"):
            if k in x and x[k] is not None:
                return str(x[k])
        return str(x)
    return str(x)


def wrap_tools_for_langchain(my_tools: Dict[str, object]) -> List[Tool]:

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

    # Web search (aliases)  <-- important: include this tool!
    if web_tool_obj is not None:
        web_fn = make_execute_fn(web_tool_obj)
        lc_tools.append(Tool(name="WebSearch", func=web_fn, description="Search the web for information."))
        lc_tools.append(Tool(name="web_search", func=web_fn, description="(alias) Search the web for information."))

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

    logger.info("[wrap_tools_for_langchain] count=%d names=%s", len(lc_tools), [t.name for t in lc_tools])
    return lc_tools


def create_ollama_llm() -> ChatOllama:
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))

    llm = ChatOllama(
        model=model,
        base_url=base_url,
        timeout=timeout,
        validate_model_on_init=True,
        disable_streaming="tool_calling",
    )

    logger.info("[llm] ChatOllama model=%s base_url=%s timeout=%s", model, base_url, timeout)
    return llm


def run_chat(agent) -> None:
    print("\nLangChain Agent Chat (create_agent).\nType 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return

        try:
            state = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            msgs = state.get("messages", [])
            final = msgs[-1].content if msgs else ""
            print(f"{final}\n")
        except Exception as e:
            logger.exception("[chat] failed")
            print(f"[ERROR] {e}\n")


def main() -> None:
    configure_logging(logging.DEBUG)

    my_tools = create_my_tools()
    tools = wrap_tools_for_langchain(my_tools)

    llm = create_ollama_llm()

    agent = create_agent(
        model=llm,
        tools=tools,
        debug=True,
    )

    run_chat(agent)


if __name__ == "__main__":
    main()
