from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections.abc import Mapping
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool

from llama_index.llms.ollama import Ollama

from evaluation.tools.calculator_tool import CalculatorTool
from evaluation.tools.vector_search_tool import VectorSearchTool
from evaluation.tools.web_search_tool import DuckDuckGoTool
from evaluation.tools.wikipedia_tool import WikipediaTool

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    log_path = Path(__file__).resolve().with_name("llamaindex_agent.log")
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
        "llama_index",
        "llama_index_instrumentation",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("[configure_logging] level=%s log_file=%s", logging.getLevelName(level), log_path)

def patch_ollama_tool_calls_none() -> None:
    """
    This function sanitises responses so tool_calls is always a list:
      None -> []
    Safe to keep even after upgrades (it becomes a harmless no-op).
    """
    try:
        import ollama 
    except Exception:
        logger.warning("[patch] ollama python client not importable; skipping patch")
        return

    def _fix_part(part):
        # part is usually a dict like:
        # {"message": {"role": "...", "content": "...", "tool_calls": None or [...]}, ...}
        if isinstance(part, dict):
            msg = part.get("message")
            if isinstance(msg, dict) and msg.get("tool_calls") is None:
                msg["tool_calls"] = []
        return part

    if hasattr(ollama, "Client") and hasattr(ollama.Client, "chat"):
        orig_chat = ollama.Client.chat

        def chat(self, *args, **kwargs):
            resp = orig_chat(self, *args, **kwargs)

            # Non-stream: a dict
            if isinstance(resp, dict):
                return _fix_part(resp)

            # Stream: an iterator yielding dicts
            if hasattr(resp, "__iter__"):
                def _gen():
                    for part in resp:
                        yield _fix_part(part)
                return _gen()

            return resp

        ollama.Client.chat = chat  # type: ignore

    if hasattr(ollama, "AsyncClient") and hasattr(ollama.AsyncClient, "chat"):
        orig_achat = ollama.AsyncClient.chat

        async def achat(self, *args, **kwargs):
            resp = await orig_achat(self, *args, **kwargs)

            # Non-stream: dict
            if isinstance(resp, dict):
                return _fix_part(resp)

            # Stream: async generator / async iterable yielding dict parts
            if hasattr(resp, "__aiter__"):
                async def _agen():
                    async for part in resp:
                        yield _fix_part(part)
                return _agen()

            return resp

        ollama.AsyncClient.chat = achat  # type: ignore

    logger.info("[patch] applied ollama tool_calls=None -> [] sanitizer")


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

    # vector search: default to bundled Austria pdf, override via VECTOR_SEARCH_FILES
    try:
        default_files = [Path(__file__).resolve().parent / "tools" / "facts_about_Austria.pdf"]
        env_files = _parse_vector_search_files_from_env()
        files = env_files if env_files else default_files
        existing = [p for p in files if p.exists()]
        if existing:
            tools["vector_search"] = VectorSearchTool(existing)
        else:
            logger.warning("[tools] vector_search disabled (no existing files). files=%s", files)
    except Exception:
        logger.exception("[tools] vector_search init failed")

    logger.info("[create_my_tools] count=%d keys=%s", len(tools), list(tools.keys()))
    return tools


_TOOL_NAME_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def _safe_tool_name(name: str) -> str:
    name = (name or "").strip() or "tool"
    name = _TOOL_NAME_RE.sub("_", name)
    return name[:64]


def _extract_text(x) -> str:
    """
    - My CalculatorTool calls .strip(), which exists on strings but NOT on dict-like
      inputs (e.g., LlamaIndex "AttributedDict"), causing:
        AttributeError: 'AttributedDict' object has no attribute 'strip'

    So the function normalises whatever LlamaIndex gives us into a string before calling tool.
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x

    if isinstance(x, Mapping):
        for k in ("input", "tool_input", "query", "expression", "text"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v

        if len(x) == 1:
            return _extract_text(next(iter(x.values())))

        try:
            return json.dumps(dict(x), ensure_ascii=False)
        except Exception:
            return str(x)

    return str(x)


def wrap_tools_for_llamaindex(my_tools: Dict[str, object]) -> List[FunctionTool]:
    llama_tools: List[FunctionTool] = []

    def make_tool_fn(tool_obj):
        # NIt accepts multiple argument names because different LlamaIndex versions
        # pass tool inputs differently. This keeps the wrapper compatible.
        def _fn(input=None, tool_input=None, query=None, expression=None, **kwargs) -> str:
            raw = input or tool_input or query or expression
            if raw is None or raw == "":
                raw = kwargs

            tool_input_str = _extract_text(raw)

            ev = tool_obj.execute(tool_input_str)

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

    for key, tool_obj in my_tools.items():
        fn = make_tool_fn(tool_obj)
        tool_name = _safe_tool_name(getattr(tool_obj, "name", key))
        tool_desc = getattr(tool_obj, "description", "") or ""

        llama_tools.append(
            FunctionTool.from_defaults(
                fn=fn,
                name=str(tool_name),
                description=str(tool_desc),
            )
        )

    logger.info("[wrap_tools_for_llamaindex] count=%d", len(llama_tools))
    return llama_tools


def create_llamaindex_ollama_llm() -> Ollama:
    model = os.getenv("OLLAMA_MODEL", "llama3:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

    # - setting function-calling OFF avoids LlamaIndex trying to parse tool_calls
    env_fc = os.getenv("LLAMAINDEX_FUNCTION_CALLING", "").strip().lower()
    if env_fc in {"1", "true", "yes", "y"}:
        is_fn_calling = True
    elif env_fc in {"0", "false", "no", "n"}:
        is_fn_calling = False
    else:
        is_fn_calling = not model.startswith("llama3")

    try:
        llm = Ollama(
            model=model,
            base_url=base_url,
            request_timeout=timeout,
            temperature=temperature,
            is_function_calling_model=is_fn_calling,
        )
    except TypeError:
        llm = Ollama(
            model=model,
            base_url=base_url,
            request_timeout=timeout,
            temperature=temperature,
        )

    logger.info(
        "[llm] LlamaIndex Ollama model=%s base_url=%s timeout=%s fn_calling=%s",
        model, base_url, timeout, is_fn_calling
    )
    return llm


async def main_async() -> None:
    configure_logging(logging.DEBUG)

    # Patch first so the LlamaIndex Ollama integration won't crash on tool_calls=None
    patch_ollama_tool_calls_none()

    logger.info("[main] init tools")
    my_tools = create_my_tools()
    llama_tools = wrap_tools_for_llamaindex(my_tools)

    llm = create_llamaindex_ollama_llm()

    agent_timeout = float(os.getenv("AGENT_TIMEOUT", "120"))
    verbose = os.getenv("AGENT_VERBOSE", "").strip().lower() in {"1", "true", "yes", "y"}

    agent = ReActAgent(
        llm=llm,
        tools=llama_tools,
        timeout=agent_timeout,
        verbose=verbose,
    )

    print(
        "\nLlamaIndex ReActAgent Chat (LlamaIndex Ollama integration + your tools).\n"
        "Type 'exit' or 'quit' to stop.\n"
        f"Model: {getattr(llm, 'model', 'unknown')}\n"
        f"Timeout: {agent_timeout}s | Verbose: {verbose}\n"
    )

    # - early_stopping_method='generate' avoids hard-failing; it tells the agent to produce
    #   a final answer even if it couldn't complete the full tool loop.
    max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "25"))
    early_stopping_method = os.getenv("AGENT_EARLY_STOPPING", "generate").strip() or "generate"

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
            resp = await agent.run(
                user_input,
                max_iterations=max_iterations,
                early_stopping_method=early_stopping_method,
            )
            print(f"{str(resp)}\n")
        except Exception as e:
            logger.exception("[chat] failed")
            print(f"[ERROR] {e}\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
