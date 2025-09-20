# source_code/chainlit_chatbot_main.py
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

import chainlit as cl

# ---- Your LLMs ----
from llm_adapters.ollama_depseek_llm import OllamaDeepseekLLM  # noqa: F401
from llm_adapters.ollama_gemma_llm import OllamaGemmaLLM
from llm_adapters.ollama_mistral_llm import OllamaMistralLLM

# ---- Your tools ----
from tool_management.tools.calculator_tool import CalculatorTool
from tool_management.tools.serpapi_tool import SerpAPITool
from tool_management.tools.wikipedia_tool import WikipediaTool
from tool_management.tools.vector_search_tool import VectorSearchTool

# ---- Agent + pronouns resolver entities ----
from agent_heart.react_agent import ReActAgent
from agent_memory.pronouns_resolution.llm_entity_extractor import LLMEntityExtractor


# ---------------- Logging (same style as main.py) ---------------- #

def configure_logging(level: int = logging.INFO) -> None:
    """Log to stdout and a rotating file next to this script."""
    here = Path(__file__).resolve()
    log_path = here.with_name("agent.log")
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
    for noisy in ("urllib3", "wikipedia", "wikipediaapi", "serpapi"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


configure_logging(logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Session keys ---------------- #

AGENT_KEY = "agent"
TOOLS_KEY = "tools"              # the shared dict passed into TaskManager/ToolExecutor
FILES_KEY = "uploaded_files"     # List[Path]
CANCEL_KEY = "cancel_event"      # asyncio.Event
TASK_KEY = "current_task"        # asyncio.Task | None


# ---------------- Builders (mirror main.py) ---------------- #

def build_llms():
    query_separator_llm = OllamaGemmaLLM()
    tool_selector_llm = OllamaGemmaLLM()
    answer_generator_llm = OllamaMistralLLM()
    confidence_assessor = OllamaGemmaLLM()
    pronouns_resolution = OllamaGemmaLLM()
    entity_extractor_llm = OllamaGemmaLLM()
    return (
        query_separator_llm,
        tool_selector_llm,
        answer_generator_llm,
        confidence_assessor,
        pronouns_resolution,
        entity_extractor_llm,
    )


def build_tools(initial_files: Optional[List[Path]] = None) -> Dict[str, object]:
    tools: Dict[str, object] = {
        "calculator": CalculatorTool(),
        "wikipedia": WikipediaTool(),
    }

    serp_key = os.getenv("SERPAPI_KEY") or "b53c73f7ff8666ff176932332e272be8175325eede05c4f1b716cbbccf6103ca"
    try:
        tools["web_search"] = SerpAPITool(api_key=serp_key)
    except Exception as e:
        logger.warning("SERPAPI init failed: %s", e)

    if initial_files:
        try:
            tools["vector_search"] = VectorSearchTool(initial_files)
        except Exception as e:
            logger.exception("Failed to build VectorSearchTool: %s", e)

    return tools


def attach_or_replace_vector_tool(tools: Dict[str, object], files: List[Path]) -> None:
    """
    Create (or replace) the vector_search tool in-place.
    Because your TaskManager/ToolExecutor keep a reference to this dict,
    updating it here makes the tool instantly available.
    """
    tools["vector_search"] = VectorSearchTool(files)
    logger.info("VectorSearchTool ready with %d file(s).", len(files))


def build_agent(initial_files: List[Path]) -> ReActAgent:
    (
        query_separator_llm,
        tool_selector_llm,
        answer_generator_llm,
        confidence_assessor_llm,
        pronouns_resolution_llm,
        entity_extractor_llm,
    ) = build_llms()

    tools = build_tools(initial_files)
    entity_extractor = LLMEntityExtractor(entity_extractor_llm)

    agent = ReActAgent(
        query_separator_llm=query_separator_llm,
        tool_selector_llm=tool_selector_llm,
        answer_generator_llm=answer_generator_llm,
        confidence_assessor_llm=confidence_assessor_llm,
        entity_extractor=entity_extractor,
        pronouns_resolution_llm=pronouns_resolution_llm,
        tools=tools,
    )
    return agent


# ---------------- Compatibility helpers ---------------- #

def _make_action_upload_button() -> cl.Action:
    """
    Chainlit's Action API changed between versions:
      - Older: Action(name=..., value=..., label=...)
      - Newer: Action(name=..., payload=..., label=...)
    We try payload first, then fall back to value.
    """
    try:
        # Newer style
        return cl.Action(
            name="upload_files",
            label="Upload/Replace files",
            payload={"action": "upload"},
        )
    except Exception:
        # Older style
        return cl.Action(
            name="upload_files",
            value="upload",
            label="Upload/Replace files",
        )


def _get_action_value_or_payload(action: cl.Action) -> Optional[str]:
    """Read either action.payload['action'] or action.value, depending on version."""
    pv = getattr(action, "payload", None)
    if isinstance(pv, dict) and "action" in pv:
        return str(pv["action"])
    return getattr(action, "value", None)


# ---------------- Chainlit UI glue ---------------- #

@cl.on_chat_start
async def on_chat_start():
    """
    1) Ask for optional files to seed VectorSearchTool.
    2) Build agent with those files.
    3) Prepare cooperative cancel primitives.
    4) Offer an action to upload/replace files later.
    """
    # Ask for files (user may skip)
    files_msg = await cl.AskFileMessage(
        content=(
            "Optional: upload PDF, DOCX, TXT, or MD files to enable vector search.\n"
            "You can also upload later using the 'Upload/Replace files' button."
        ),
        accept=[".pdf", ".docx", ".txt", ".md"],
        max_size_mb=20,
        max_files=6,
        timeout=60,
    ).send()

    uploaded: List[Path] = []
    if files_msg:
        for f in files_msg:
            uploaded.append(Path(f.path))

    # Build & store agent
    agent = build_agent(uploaded)
    cl.user_session.set(AGENT_KEY, agent)
    cl.user_session.set(TOOLS_KEY, agent.task_manager.tools)  # shared dict
    cl.user_session.set(FILES_KEY, uploaded)

    # Create per-session cancel event + current task holder
    cl.user_session.set(CANCEL_KEY, asyncio.Event())
    cl.user_session.set(TASK_KEY, None)

    # Welcome message + action (compat helper to avoid pydantic ValidationError)
    await cl.Message(
        content=(
            "👋 Ready! Ask me anything.\n\n"
            "• If you uploaded files, try: *“summarize the uploaded documents”* or "
            "*“find sections about ReAct agents”*.\n"
            "• You can replace files anytime:"
        ),
        actions=[_make_action_upload_button()],
    ).send()


@cl.on_stop
async def on_stop():
    """Triggered when the user presses the Stop button in Chainlit."""
    ev: asyncio.Event = cl.user_session.get(CANCEL_KEY)
    if ev:
        ev.set()
    task: Optional[asyncio.Task] = cl.user_session.get(TASK_KEY)
    if task and not task.done():
        task.cancel()


@cl.action_callback("upload_files")
async def on_upload_action(action: cl.Action):
    """Let user upload/replace files at any time."""
    # Ignore unknown actions gracefully (compat across versions)
    act = _get_action_value_or_payload(action)
    if act not in (None, "upload"):
        return

    ask = await cl.AskFileMessage(
        content="Upload files to (re)build the semantic index.",
        accept=[".pdf", ".docx", ".txt", ".md"],
        max_size_mb=20,
        max_files=10,
        timeout=120,
    ).send()

    files: List[Path] = []
    if ask:
        files = [Path(f.path) for f in ask]

    # Update session state & tools
    cl.user_session.set(FILES_KEY, files)
    tools: Dict[str, object] = cl.user_session.get(TOOLS_KEY)  # shared tools dict
    try:
        attach_or_replace_vector_tool(tools, files)
        await cl.Message(
            content=f"Vector index rebuilt with {len(files)} file(s). You can now query them."
        ).send()
    except Exception as e:
        logger.exception("Vector tool rebuild failed: %s", e)
        await cl.Message(content=f"Failed to build vector index: {e}").send()


@cl.on_message
async def on_message(msg: cl.Message):
    """
    Run the agent with cooperative cancellation.
    Also index any files dragged directly into this message.
    """
    agent: ReActAgent = cl.user_session.get(AGENT_KEY)
    if not agent:
        await cl.Message(content="Agent not ready. Please refresh.").send()
        return

    # If this message contains fresh files, (re)build vector tool.
    new_paths: List[Path] = []
    for elem in msg.elements or []:
        if getattr(elem, "path", None):
            new_paths.append(Path(elem.path))

    if new_paths:
        tools: Dict[str, object] = cl.user_session.get(TOOLS_KEY)
        try:
            attach_or_replace_vector_tool(tools, new_paths)
            cl.user_session.set(FILES_KEY, new_paths)
            await cl.Message(content=f"Indexed {len(new_paths)} new file(s).").send()
        except Exception as e:
            logger.exception("Inline file indexing failed: %s", e)
            await cl.Message(content=f"Failed to index uploaded files: {e}").send()

    # Cooperative cancel primitives
    ev: asyncio.Event = cl.user_session.get(CANCEL_KEY)
    if ev:
        ev.clear()

    async def run_agent() -> str:
        """
        Prefer a cooperative method if your agent implements it, otherwise fall back.
        Implement `handle_with_cancel(self, query: str, cancel_event: asyncio.Event) -> str`
        in your ReActAgent for true cancellation.
        """
        if hasattr(agent, "handle_with_cancel"):
            # type: ignore[attr-defined]
            return await asyncio.to_thread(agent.handle_with_cancel, msg.content.strip(), ev)
        # Fallback: run normal handle (can't actually interrupt the underlying work)
        return await asyncio.to_thread(agent.handle, msg.content.strip())

    task = asyncio.create_task(run_agent())
    cl.user_session.set(TASK_KEY, task)

    try:
        answer = await task
        await cl.Message(content=answer).send()
    except asyncio.CancelledError:
        await cl.Message(content="🛑 Stopped.").send()
    finally:
        cl.user_session.set(TASK_KEY, None)
