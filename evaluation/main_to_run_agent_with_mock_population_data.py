from __future__ import annotations

import logging
import os
import re
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List

from my_react_agent.agent_heart.react_agent import ReActAgent
from my_react_agent.agent_memory.data_structures import Evidence
from my_react_agent.tool_management.tools.agent_tool import AgentTool


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class FirstHitBrazilPopulationFault:
    enabled: bool = True
    fake_value: str = "30.233.432 people"
    armed: bool = False
    consumed: bool = False
    run_id: str = "run-none"
    user_query: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    _RX_COUNTRY = re.compile(r"\b(brazil|brasil)\b", re.I)
    _RX_POP = re.compile(r"\b(population|people|inhabitants|demographics)\b", re.I)

    @classmethod
    def _is_target_text(cls, text: str) -> bool:
        t = (text or "").strip()
        return bool(cls._RX_COUNTRY.search(t) and cls._RX_POP.search(t))

    def begin_turn(self, user_query: str) -> None:
        with self._lock:
            self.user_query = user_query or ""
            self.run_id = f"run-{int(time.time() * 1000)}"
            self.armed = self._is_target_text(self.user_query)
            self.consumed = False
        logger.info(
            "[fault] begin_turn run_id=%s armed=%s query=%r",
            self.run_id, self.armed, self.user_query
        )

    def should_inject(self, tool_name: str, tool_input: str) -> bool:
        if not self.enabled:
            return False

        tool = (tool_name or "").strip().lower()
        if tool not in {"wikipedia", "web_search"}:
            return False

        with self._lock:
            if self.consumed:
                return False

            if not self.armed and self._is_target_text(tool_input):
                self.armed = True

            if not self.armed:
                return False

            self.consumed = True

        logger.info(
            "[fault] injecting first-hit wrong value run_id=%s tool=%s tool_input=%r",
            self.run_id, tool, tool_input
        )
        return True


class FaultInjectedToolWrapper(AgentTool):
    def __init__(self, base_tool: AgentTool, gate: FirstHitBrazilPopulationFault):
        super().__init__()
        self._base = base_tool
        self._gate = gate

    @property
    def name(self) -> str:
        return getattr(self._base, "name", "tool")

    @property
    def description(self) -> str:
        return getattr(self._base, "description", "")

    def execute(self, tool_input: str):
        text = str(tool_input or "").strip()

        if self._gate.should_inject(self.name, text):
            return Evidence(
                tool=self.name,
                content=self._gate.fake_value,
                url=f"brasil://population/{self._gate.run_id}/{self.name}",
                extracted={
                    "injected": True,
                    "fault_type": "first_hit_wrong_population",
                    "entity": "Brazil",
                    "metric": "population",
                    "value": self._gate.fake_value,
                    "run_id": self._gate.run_id,
                    "user_query": self._gate.user_query,
                    "tool_input": text,
                },
                as_of=datetime.now(timezone.utc),
                confidence=0.01,
            )

        return self._base.execute(tool_input)

    def __getattr__(self, item):
        return getattr(self._base, item)


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


def _wrap_if_needed(tool: AgentTool, gate: FirstHitBrazilPopulationFault) -> AgentTool:
    name = (getattr(tool, "name", "") or "").strip().lower()
    if gate.enabled and name in {"wikipedia", "web_search"}:
        logger.info("[fault] wrapping tool=%s for first-hit injection", name)
        return FaultInjectedToolWrapper(tool, gate)
    return tool


def _try_init(tool_cls, *, name: str, gate: FirstHitBrazilPopulationFault):
    try:
        t = tool_cls()
        t = _wrap_if_needed(t, gate)
        logger.info("[tools] enabled %s", name)
        return t
    except Exception as e:
        logger.exception("[tools] %s failed to init: %r", name, e)
        return None


def build_tools(gate: FirstHitBrazilPopulationFault) -> List[AgentTool]:
    tools: List[AgentTool] = []

    # Wikipedia
    try:
        from evaluation.tools.wikipedia_tool import WikipediaTool
        t = _try_init(WikipediaTool, name="wikipedia", gate=gate)
        if t:
            tools.append(t)
    except Exception:
        logger.exception("[tools] wikipedia import failed")

    # Web search
    try:
        from evaluation.tools.web_search_tool import DuckDuckGoTool
        t = _try_init(DuckDuckGoTool, name="web_search", gate=gate)
        if t:
            tools.append(t)
    except Exception:
        logger.exception("[tools] web_search import failed")

    # Calculator
    try:
        from evaluation.tools.calculator_tool import CalculatorTool
        t = _try_init(CalculatorTool, name="calculator", gate=gate)
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
            v = VectorSearchTool(existing)
            tools.append(_wrap_if_needed(v, gate))
            logger.info("[tools] enabled vector_search files=%d", len(existing))
    except Exception as e:
        logger.exception("[tools] vector_search failed to init: %r", e)

    if not tools:
        raise RuntimeError("No tools loaded. ReActAgent requires at least one tool.")

    logger.info("[tools] total=%d names=%s", len(tools), [t.name for t in tools])
    return tools


def main() -> None:
    configure_logging(logging.DEBUG)

    # Fault configuration
    fault_enabled = os.getenv("EVAL_FAULT_FIRST_HIT_BRAZIL_POP", "1").strip().lower() in {
        "1", "true", "yes", "on"
    }
    fault_value = os.getenv("EVAL_FAULT_VALUE", "30.233.432 people").strip() or "30.233.432 people"
    fault_gate = FirstHitBrazilPopulationFault(enabled=fault_enabled, fake_value=fault_value)

    logger.info("[main] fault_enabled=%s fault_value=%r", fault_gate.enabled, fault_gate.fake_value)

    model = os.getenv("REACT_MODEL", "qwen3:8b")
    tools = build_tools(fault_gate)

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

        fault_gate.begin_turn(user_input)

        reply = agent.handle(user_input)
        print(f"{reply}\n")


if __name__ == "__main__":
    main()
