from __future__ import annotations

import ast
import importlib
import logging
import operator as op
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional

from my_react_agent.agent_core.agent_actions import (
    AnswerByItselfAction,
    ClarifyAction,
    UseToolAction,
    StopAction,
)
from my_react_agent.agent_core.agent_actions.need_context_action import NeedContextAction
from my_react_agent.agent_heart.react_agent import ReActAgent, ParameterAssessorFactory
from my_react_agent.agent_memory.llm_entity_extractor import LLMEntityExtractor
from my_react_agent.agent_prompts.prompt_registry import PromptRegistry
from my_react_agent.agent_prompts.defaults_prompts import DEFAULT_PROMPTS

import ollama
from my_react_agent.llm_adapters.llm_base import LLMBase

from my_react_agent.tool_management.tools.agent_tool import AgentTool

from my_react_agent.agent_memory.data_structures import Evidence

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OllamaGemmaLLM(LLMBase):
    def __init__(self, model: str = "gemma3:4b", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_ctx": kwargs.get("num_ctx", 2048),
                    "stop": kwargs.get("stop", ["<|endoftext|>"])
                }
            )
            return response['response']
        except Exception as e:
            raise ConnectionError(f"Ollama error: {str(e)}")

# ----------------------------
# Safe calculator implementation
# ----------------------------
_ALLOWED_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_ALLOWED_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval")

    def _eval(n: ast.AST):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BIN_OPS:
            return _ALLOWED_BIN_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARY_OPS:
            return _ALLOWED_UNARY_OPS[type(n.op)](_eval(n.operand))
        raise ValueError(f"Unsupported expression element: {type(n).__name__}")

    return float(_eval(node))


def _extract_math_expression(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip()

    # Try to grab a substring that looks like arithmetic.
    m = re.search(r"[0-9\.\s\+\-\*\/\(\)\%\^]+", s)
    if not m:
        return None
    expr = m.group(0).strip()
    expr = expr.replace("^", "**")  # convenience
    return expr or None


class CalculatorTool(AgentTool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Safely evaluates basic arithmetic expressions (e.g. 2+2, (3.5*4)-1, 2^10)."

    @property
    def refiner_instructions(self) -> str:
        return "Output only a valid arithmetic expression using digits and operators + - * / // % ** ( ) ."

    @property
    def refiner_input_format(self) -> str:
        return "<arithmetic expression>"

    @property
    def refiner_input_regex(self) -> Optional[str]:
        return r"^[0-9\.\s\+\-\*\/\(\)\%\^]+$"

    def execute(self, input: str) -> Evidence:
        q = (input or "").strip()
        expr = _extract_math_expression(q)

        if not expr:
            return Evidence(
                tool=self.name,
                content="Calculator failed: no arithmetic expression found.",
                url=None,
                extracted={"error": True, "message": "no_expression", "input": q},
                as_of=datetime.utcnow(),
                confidence=0.2,
            )

        try:
            value = _safe_eval(expr)
            if abs(value - round(value)) < 1e-12:
                rendered = str(int(round(value)))
            else:
                rendered = str(value)

            return Evidence(
                tool=self.name,
                content=f"{expr} = {rendered}",
                url=None,
                extracted={"expression": expr, "result": value},
                as_of=datetime.utcnow(),
                confidence=0.95,
            )
        except Exception as e:
            return Evidence(
                tool=self.name,
                content=f"Calculator failed: {e}",
                url=None,
                extracted={"error": True, "message": str(e), "expression": expr, "input": q},
                as_of=datetime.utcnow(),
                confidence=0.1,
            )

def configure_logging(level: int = logging.INFO) -> None:
    log_path = Path(__file__).resolve().with_name("agent.log")
    handlers = [
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(str(log_path), maxBytes=2_000_000, backupCount=2, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )

def create_prompt_registry() -> PromptRegistry:
    return PromptRegistry(_defaults=dict(DEFAULT_PROMPTS))


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
            logger.info("enabled assessor %s.%s", mod_path, cls_name)
        except Exception as e:
            logger.info("assessor %s.%s not available (skipping): %r", mod_path, cls_name, e)
    return factories


def main() -> None:
    configure_logging(logging.INFO)

    prompts = create_prompt_registry()

    # Minimal tool set: just calculator (ReActAgent needs at least one tool)
    tools: Dict[str, object] = {"calculator": CalculatorTool()}

    # Use Gemma for all roles (simple QuickStart)
    llms = {
        "planner_llm": OllamaGemmaLLM(),
        "summariser_llm": OllamaGemmaLLM(),
        "confidence_llm": OllamaGemmaLLM(),
        "refiner_llm": OllamaGemmaLLM(),
    }

    entity_extractor = LLMEntityExtractor(llms["summariser_llm"], prompts=prompts)

    # Same actions as your main
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

    agent = ReActAgent(
        planner_llm=llms["planner_llm"],
        summariser_llm=llms["summariser_llm"],
        confidence_llm=llms["confidence_llm"],
        refiner_llm=llms["refiner_llm"],
        entity_extractor=entity_extractor,
        tools=tools,
        prompts=prompts,
        max_steps=8,
        step_actions=step_actions,
        low_conf_actions=low_conf_actions,
        parameter_assessors=parameter_assessors,
    )

    print(
        "\nQuickStart ReAct Agent (Gemma via Ollama)\n"
        "Try: 'What is (23*7) - 5?' or 'Compute 2^10'\n"
        "Type 'exit' to quit.\n"
    )

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        reply = agent.handle(user_input)
        print(f"{reply}\n")


if __name__ == "__main__":
    main()