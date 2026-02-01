# my-react-agent

A **ReAct (Reason + Act)** agent framework for Python with **step-by-step traceability**, **evidence-first answering**, and **confidence-gated retries**.  
It plans a multi-step solution, executes each step via actions/tools, evaluates quality, and produces a final answer **grounded in collected observations and evidence**.

---

## Why this project

Similar projects- LangChain and LlamaIndex are strong frameworks—but they’re optimised for *different priorities*:

- **LangChain** is an **integration + composition** system (chains, agents, tool wrappers, retrievers, many providers). It’s great when you want to assemble an app quickly from lots of building blocks.
- **LlamaIndex** is a **data/RAG framework** (ingestion, indexing, retrieval, routing, structured querying). It’s great when your core problem is “connect LLMs to your data” at scale.

`my-react-agent` exists for a different goal: **a small, inspectable agent runtime where traceability, evidence and reliability are first-class — not optional add-ons**.

### What you get here that’s harder to guarantee in LangChain/LlamaIndex

- **Traceability as a core invariant (not a plugin / external service dependency)**  
  Every step *must* produce a structured record: action decision → tool input/output → observation → evidence → confidence.  
  This makes debugging and evaluation predictable because the “paper trail” is built into the runtime.

- **Evidence-first answering as a default design**  
  The final answer is synthesised from collected **observations + `Evidence` objects**, making it straightforward to enforce “don’t invent facts” policies and to display citations/snippets in a consistent format.

- **Confidence-gated retries with a controlled recovery loop**  
  Low-confidence step results trigger a deterministic retry policy (switch action/tool, adjust input, or stop/clarify).  
  Many frameworks can *evaluate*, but `my-react-agent` treats step-level confidence as an orchestration primitive.

- **Cleaner extension points for research/prototyping**  
  Instead of customising a big graph of components, you can add a new behavior by implementing:
  - an `Action` (LLM-visible selection rule + instructions)
  - an `ActionHandler` (runtime execution)
  
  This makes it easier to experiment with new “agent behaviors” (like GreetingAction, guardrails, special routing) without rewriting the core loop.

### When `my-react-agent` is the better choice

Use this project when you care most about:
- **auditing** (exactly what happened and why, step-by-step),
- **reproducible debugging** (structured traces you can log or test),
- **grounded outputs** (final answer constrained to collected evidence),
- **reliability under uncertainty** (confidence gating + retries),
- **lightweight core** (clear orchestration over large ecosystem complexity).

---

## Key features

- **Plan → Execute → Finalise pipeline**  
  Creates a step plan, runs each step deterministically, then synthesises a final answer.
- **Explicit traceability**  
  Step transcript + evidence pack per step (what happened, why, and what was found).
- **Evidence-first design**  
  Uses structured `Evidence` objects; final answer can be constrained to what was observed.
- **Confidence gating + retry loops**  
  Evaluates each step (alignment/quality/realism) and retries when confidence is below threshold.
- **Pluggable tools**  
  Tools are registered once and invoked through a single boundary (`ToolExecutor` / tool interface).
- **Modular actions**  
  Actions like `USE_TOOL`, `ANSWER_BY_ITSELF`, `CLARIFY`, `STOP`, and `NEED_CONTEXT` are isolated modules.
- **Memory**  
  `QueryMemory` (per question) + `ConversationMemory` (cross-turn) for entities, steps, and observations.
- **Prompt registry**  
  Centralised prompt management (`PromptRegistry`) with overridable defaults.
- **Plugin support**  
  Optional runtime extension via `REACT_AGENT_PLUGINS`.

---

## License
MIT

## Requirements
- Python 3.10+
- Ollama (local LLM runtime)

### From PyPI
pip install my-react-agent

## From Source
pip install git+https://git01lab.cs.univie.ac.at/zhaniyaa77/my-react-agent.git

## Install Ollama
#### Download and install Ollama:
- https://ollama.com/download

#### Pull a model (example used below: llama3):
ollama pull llama3

## Quickstart

To run the agent you need two things:
1. At least one tool (otherwise the agent will show error)
2. An LLM model (so the agent can reason and decide when/how to use tools)
This example uses a minimal tool called multiplier that multiplies two numbers. 

- A tool should contain methods for refiner to get structured input. The tool also defines a regex (refiner_input_regex) to validate refined inputs before execution. This helps prevent accidental tool calls on malformed strings and keeps behavior predictable. 

The refiner is a prompt-based step that converts natural language into the exact format the tool expects. 
That's why in these example you can run not just "4*5" query but also "calculate 4*5"/"what is 4 times 5?"/"multiply 4 and 5".

- Pull the model used in this example:
ollama pull llama3:8b

```python
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

import ollama

from my_react_agent.agent_memory.data_structures import Evidence
from my_react_agent.tool_management.tools.agent_tool import AgentTool

from my_react_agent.agent_heart.react_agent import ReActAgent


class MultiplicationTool(AgentTool):
    @property
    def name(self) -> str:
        return "multiplier"

    @property
    def description(self) -> str:
        return "Multiplies two numbers. Tool input must be '<number>*<number>' (e.g. 45*7)."

    # Refiner contract: makes tool input predictable
    @property
    def refiner_instructions(self) -> str:
        return "Output ONLY '<number>*<number>' (example: 45*7). No words."

    @property
    def refiner_input_format(self) -> str:
        return "<number>*<number>"

    @property
    def refiner_input_regex(self) -> Optional[str]:
        # Keep it strict so the refiner can be validated before tool execution
        return r"^\s*-?\d+(\.\d+)?\s*\*\s*-?\d+(\.\d+)?\s*$"

    def execute(self, input: str) -> Evidence:
        s = (input or "").strip()
        try:
            left, right = s.split("*", 1)      
            result = float(left) * float(right)  

            rendered = str(int(result)) if abs(result - round(result)) < 1e-12 else str(result)

            return Evidence(
                tool=self.name,
                content=f"{left.strip()} * {right.strip()} = {rendered}",
                url=None,
                extracted={"expression": s, "result": result},
                as_of=datetime.utcnow(),
                confidence=0.95,
            )
        except Exception as e:
            # Minimal fallback if refiner didn’t behave / user input slipped through
            return Evidence(
                tool=self.name,
                content=f"Multiply failed: expected '<number>*<number>' (e.g. 45*7). Error: {e}",
                url=None,
                extracted={"error": True, "input": s},
                as_of=datetime.utcnow(),
                confidence=0.1,
            )



def main() -> None:
    tools = [MultiplicationTool()]

    agent = ReActAgent.create(llm="llama3:8b", tools=tools)

    print("\nQuickStart: my-react-agent (minimal)\nType 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        print(agent.handle(q))
        print()

if __name__ == "__main__":
    main()
```

## If you want to disable logger messages:
- Do this at the very top of main() (before anything imports/initializes logging):

```python
import logging

def disable_all_logs() -> None:
    # Stops everything routed through Python logging
    logging.disable(logging.CRITICAL)

    # Also silence common chatty libraries explicitly (extra-safe)
    for name in ("httpx", "httpcore", "ollama"):
        logging.getLogger(name).disabled = True

# call first
disable_all_logs()
```

## Architecture (precise)

### High-level flow

1. **Planning (planner LLM)**
   - Input: user question (+ optional conversation state)
   - Output: one or more **step tasks** (plan)

2. **Execution loop (per step)**
   - Select an **action** (e.g. `USE_TOOL`, `ANSWER_BY_ITSELF`, `CLARIFY`, `NEED_CONTEXT`, `STOP`)
   - If tool is needed:
     - Optional **tool query refinement** produces strict tool input
     - Execute tool
   - Save **observation** + **evidence** to memory

3. **Confidence assessment**
   - Parameter assessors score the step (e.g. entity alignment, answer quality, realism)
   - If confidence < threshold → recovery loop chooses a better next action

4. **Finalisation (summariser LLM)**
   - Synthesises a final answer from step observations/evidence.

### Component map (modules and responsibilities)

**Core orchestration**
- `ReActAgent`  
  Owns the plan/execute/finalise loop, action selection, retries, and memory writes.

**Actions (step-level behaviours)**
- `NeedContextAction`  
  Resolves missing entities / pronouns using the entity extractor and memory.
- `UseToolAction`  
  Invokes exactly one tool (via the tool execution boundary), stores observation/evidence.
- `AnswerByItselfAction`  
  Uses LLM-only knowledge for stable facts (no tools).
- `ClarifyAction`  
  Asks a single clarification question when the step is underspecified.
- `StopAction`  
  Terminates after repeated failures or user cancellation.

**Tools**
- `AgentTool` (interface/base class)  
  Tools implement `execute(tool_input: str) -> Evidence`.
- `ToolExecutor` (execution boundary)  
  The only place where the agent invokes tools. Keeps tool I/O consistent and traceable.

**Memory**
- `QueryMemory`  
  Per-question state: plan, step trace, transcript, observations.
- `ConversationMemory`  
  Cross-question state: extracted entities and references you want to persist.

**Evidence**
- `Evidence` (structured record)  
  `tool`, `content`, `url`, `extracted` dict, `as_of`, `confidence`.

**Confidence**
- Parameter assessors (factory-driven)  
  Examples: `EntityAlignmentAssessor`, `AnswerQualityAssessor`, `AnswerRealismAssessor`.

**Tool input refinement (ToolQueryRefiner)**
- Before calling a tool, the agent converts current step's task into the exact tool input string expected by that tool. This is what prevents “LLM prose” from being fed into tools and standardizes tool calls.
- ToolQueryRefiner relies on AgentTool exposing a “refiner contract”. Tools can implement these properties to constrain/refine the model’s output:
`refiner_instructions`: str — tool-specific rules (“Return a normal search query…”, etc.)
`refiner_input_format`: str — short format spec for expected input
`refiner_input_regex`: Optional[str] — strict regex for allowed inputs
`refiner_forbidden`: str — explicit forbidden patterns
`refiner_examples`: str / get_examples() — optional examples to guide the refiner
`refiner_max_chars`: int — max tool input length (hard cap)

**Prompts**
- `PromptRegistry`  
  Stores prompt templates for planning, refinement, confidence assessment, summarisation.

**Plugins**
- Loaded via `REACT_AGENT_PLUGINS` environment variable  
  A plugin module exposes `plugin.register(ctx)` and can add tools/actions/assessors/prompts.


## Adding a Custom Action
- Adding a Custom Action (Example: GreetingAction)

- Goal: If the user starts with a greeting (e.g., “hi”, “hello”, “good morning”), the agent should include a greeting back in the final answer.


```python
from __future__ import annotations

"""

This code does:
1) Define a new Action (metadata the planner LLM sees).
2) Define a new ActionHandler (runtime logic).
3) Register the action in step_actions.
4) Register the handler via ReActAgent(action_handlers=...) (no core edits needed).
5) Run the agent.

Prereqs:
- Ollama installed + running
- Model pulled (default below):  ollama pull gemma3:4b

Run:
    python main_custom_action_demo.py
"""

import ast
import importlib
import logging
import operator as op
import re
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import ollama

from my_react_agent.llm_adapters.llm_base import LLMBase
from my_react_agent.agent_prompts.prompt_registry import PromptRegistry
from my_react_agent.agent_prompts.defaults_prompts import DEFAULT_PROMPTS

from my_react_agent.agent_core.agent_actions.action import Action
from my_react_agent.agent_core.agent_actions import (
    AnswerByItselfAction,
    ClarifyAction,
    UseToolAction,
    StopAction,
)
from my_react_agent.agent_core.agent_actions.need_context_action import NeedContextAction

from my_react_agent.agent_heart.react_agent import ReActAgent, ParameterAssessorFactory
from my_react_agent.agent_heart.action_handler_base import ActionHandler, empty_tool_call
from my_react_agent.agent_memory.llm_entity_extractor import LLMEntityExtractor

from my_react_agent.tool_management.tools.agent_tool import AgentTool
from my_react_agent.agent_memory.data_structures import Evidence, Step, StepResult, StepToolCall, step_set_result


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 1) LLM adapter (Ollama ->  LLMBase)
class OllamaGemmaLLM(LLMBase):
    def __init__(self, model: str = "gemma3:4b", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs) -> str:
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": self.temperature,
                "num_ctx": kwargs.get("num_ctx", 2048),
                "stop": kwargs.get("stop", ["<|endoftext|>"]),
            },
        )
        return response["response"]

# 2) Minimal Calculator tool (so the agent has at least one tool)

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
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BIN_OPS:
            return _ALLOWED_BIN_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARY_OPS:
            return _ALLOWED_UNARY_OPS[type(n.op)](_eval(n.operand))
        raise ValueError(f"Unsupported expression element: {type(n).__name__}")

    return float(_eval(node))


def _extract_math_expression(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"[0-9\.\s\+\-\*\/\(\)\%\^]+", text.strip())
    if not m:
        return None
    expr = m.group(0).strip().replace("^", "**")
    return expr or None


class CalculatorTool(AgentTool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Safely evaluates basic arithmetic expressions."

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
            rendered = str(int(value)) if abs(value - round(value)) < 1e-12 else str(value)
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


# 3) Custom Action: GreetingAction (what the planner LLM can choose)

class GreetingAction(Action):
    """
    This is the "catalogue entry" the planner sees.
    DECIDE_ACTION_FOR_STEP prompt will list this action with:
    - name
    - when_to_pick
    - instructions
    - examples (optional)

    IMPORTANT:
    - name must match the handler registration key ("GREETING" below).
    """

    @property
    def name(self) -> str:
        return "GREETING"

    @property
    def default_when_to_pick(self) -> str:
        return (
            "Pick when the user message starts with a greeting (hi/hello/hey/good morning/etc.) "
            "and we should greet back."
        )

    @property
    def default_instructions(self) -> str:
        return (
            "If the user starts with a greeting, create a short friendly greeting. "
            "Do NOT answer the main question here; only add greeting context."
        )

    @property
    def examples(self) -> list[str]:
        return [
            "User: Hi! What is the capital of Germany?",
            "User: Hello, can you explain ReAct agents?",
            "User: Good morning — what is the weather in Tokyo?",
        ]


# 4) Custom Handler: GreetingHandler (runtime logic when GREETING is selected)
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening)\b",
    flags=re.I,
)


class GreetingHandler(ActionHandler):
    """
    Runtime behavior:
    - Detect greeting from the original user question
    - Store a context snippet so it shows up in final synthesis
    - Return a short observation so the transcript shows what happened
    """

    @property
    def action_name(self) -> str:
        return "GREETING"

    def run(self, ctx) -> Tuple[StepToolCall, StepResult, Step]:
        user_text = (ctx.question or "").strip()

        greeting_text = ""
        if _GREETING_RE.search(user_text):
            greeting_text = "Hello! 👋"
            # This is the easiest way to make final synthesis "see" the greeting.
            ctx.agent._context_snippets.append(f"GREETING: {greeting_text}")

        observation = greeting_text or "No greeting detected."
        step_result = StepResult(
            observation=observation,
            final_answer=None,
            should_stop=False,
            success=True,
        )

        # Optional: attach traceable evidence
        ev = Evidence(
            tool="greeting",
            content=observation,
            url=None,
            extracted={"greeting": greeting_text, "matched": bool(greeting_text)},
            as_of=datetime.utcnow(),
            confidence=0.9,
        )

        step = ctx.step
        try:
            if getattr(step, "evidence", None) is not None:
                step.evidence.append(ev)
        except Exception:
            pass

        updated_step = step_set_result(step, step_result)
        return empty_tool_call(tool=""), step_result, updated_step

    def should_assess_result(self, ctx, *, step: Step, decision, step_result: StepResult) -> bool:
        # This is deterministic; don't waste confidence budget / retries on it.
        return False


# 5) Setup helpers (logging, prompts, assessors)

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


# 6) Main

def main() -> None:
    configure_logging(logging.INFO)

    prompts = create_prompt_registry()

    # Tool set (minimal)
    tools: Dict[str, object] = {"calculator": CalculatorTool()}

    # One model for all roles (QuickStart style)
    llms = {
        "planner_llm": OllamaGemmaLLM(),
        "summariser_llm": OllamaGemmaLLM(),
        "confidence_llm": OllamaGemmaLLM(),
        "refiner_llm": OllamaGemmaLLM(),
    }

    entity_extractor = LLMEntityExtractor(llms["summariser_llm"], prompts=prompts)

    # Register custom action FIRST so the planner sees it early in the catalogue.
    step_actions = [
        GreetingAction(),
        NeedContextAction(),
        AnswerByItselfAction(),
        ClarifyAction(),
        UseToolAction(),
        StopAction(),
    ]

    # Low confidence actions can include it too, but it's optional.
    low_conf_actions = [
        GreetingAction(),
        NeedContextAction(),
        UseToolAction(),
        AnswerByItselfAction(),
        StopAction(),
        ClarifyAction(),
    ]

    parameter_assessors = create_parameter_assessor_factories()

    # Key part: register handler WITHOUT editing ReActAgent core
    action_handlers = {
        "GREETING": GreetingHandler(),
    }

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
        action_handlers=action_handlers,  # <-- this wires GREETING -> GreetingHandler()
    )

    print(
        "\nCustom Action Demo: GREETING\n"
        "Try:\n"
        "  Hi! What is (23*7) - 5?\n"
        "  hello compute 2^10\n"
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
```

## Adding a Custom Tool

In `my-react-agent`, a **tool** is any component that implements the `AgentTool` interface:

- **Input**: a single `str` (`tool_input`)
- **Output**: an `Evidence` object (structured, traceable, timestamped)

Tools are executed through a single boundary (`ToolExecutor`) and are typically triggered by the `USE_TOOL` action (via `UseToolAction`).

This section shows how to add a new tool: a **tiny calculator** that reads an image from disk and returns structured evidence.

```python
from __future__ import annotations

"""
What this does:
1) Define a Tool class implementing AgentTool
2) Register it in `tools = {...}`
3) Run ReActAgent so the planner can pick USE_TOOL and call tool

Prereqs:
- Ollama installed + running
- Pull model:  ollama pull gemma3:4b

Run:
    python main_custom_tool_demo.py

Try:
    "Compute 2+2"
    "What is (23*7) - 5?"
    "Use the calculator to do 2^10"
"""

import importlib
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional

import ollama

from my_react_agent.llm_adapters.llm_base import LLMBase
from my_react_agent.agent_prompts.prompt_registry import PromptRegistry
from my_react_agent.agent_prompts.defaults_prompts import DEFAULT_PROMPTS

from my_react_agent.agent_core.agent_actions import (
    AnswerByItselfAction,
    ClarifyAction,
    UseToolAction,
    StopAction,
)
from my_react_agent.agent_core.agent_actions.need_context_action import NeedContextAction

from my_react_agent.agent_heart.react_agent import ReActAgent, ParameterAssessorFactory
from my_react_agent.agent_memory.llm_entity_extractor import LLMEntityExtractor

from my_react_agent.tool_management.tools.agent_tool import AgentTool
from my_react_agent.agent_memory.data_structures import Evidence


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 1) Minimal LLM adapter (Ollama -> LLMBase)
class OllamaGemmaLLM(LLMBase):
    def __init__(self, model: str = "gemma3:4b", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs) -> str:
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": self.temperature,
                "num_ctx": kwargs.get("num_ctx", 2048),
                "stop": kwargs.get("stop", ["<|endoftext|>"]),
            },
        )
        return response["response"]


# 2) The Custom Tool 
class TinyCalculatorTool(AgentTool):
    """
    - This uses eval() for demo simplicity — do NOT use eval in production.
      For production, copy Safe AST calculator from QuickStart instead.
    """

    @property
    def name(self) -> str:
        # This is the string the planner will put into decision.tool_name
        return "tiny_calculator"

    @property
    def description(self) -> str:
        # The planner sees this and decides whether to pick USE_TOOL + this tool.
        return "Evaluates a simple math expression like 2+2 or (3*7)-1. Input should be just the expression."

    # ToolQueryRefiner hints (optional but helps the planner/refiner a lot)
    @property
    def refiner_instructions(self) -> str:
        return "Return ONLY the math expression. No words. Examples: 2+2 ; (23*7)-5 ; 2**10"

    @property
    def refiner_input_format(self) -> str:
        return "<math expression>"

    @property
    def refiner_input_regex(self) -> Optional[str]:
        # Keep it simple: numbers, whitespace, and common operators/parentheses.
        return r"^[0-9\.\s\+\-\*\/\(\)\%]+$"

    def execute(self, input: str) -> Evidence:
        # 1–2 line demo logic (again: eval is for demo only)
        result = eval((input or "").replace("^", "**"), {"__builtins__": {}})  # noqa: S307

        return Evidence(
            tool=self.name,
            content=f"{input} = {result}",
            url=None,
            extracted={"expression": input, "result": result},
            as_of=datetime.utcnow(),
            confidence=0.9,
        )


# 3) Helpers (logging, prompts, assessors)
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


# 4) Main: register tool -> build agent -> chat
def main() -> None:
    configure_logging(logging.INFO)

    prompts = create_prompt_registry()

    # THIS IS THE ENTIRE "REGISTER TOOL" STEP:
    tools: Dict[str, object] = {
        "tiny_calculator": TinyCalculatorTool(),
        # add more tools here later from eva
    }

    llms = {
        "planner_llm": OllamaGemmaLLM(),
        "summariser_llm": OllamaGemmaLLM(),
        "confidence_llm": OllamaGemmaLLM(),
        "refiner_llm": OllamaGemmaLLM(),
    }

    entity_extractor = LLMEntityExtractor(llms["summariser_llm"], prompts=prompts)

    step_actions = [
        NeedContextAction(),
        AnswerByItselfAction(),
        ClarifyAction(),
        UseToolAction(),  # <-- this is what triggers tool usage
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
        "\nCustom Tool Demo: tiny_calculator\n"
        "Try:\n"
        "  Compute 2+2\n"
        "  What is (23*7) - 5?\n"
        "  Use the calculator to do 2^10\n"
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
```

#### How the agent decides to call the tool
- The planner LLM sees each tool’s:
1. name
2. description
- When it chooses USE_TOOL, it outputs a tool_name and the refiner produces tool_input.

- To make a tool easier to select:
1. Use a very specific description
2. Provide strict refiner_instructions and refiner_input_regex
3. Keep the input format simple (path:... question:...)


## Adding a Custom ParameterAssessor (Example: `RelevanceAssessor`)

This section shows how to add a new **ParameterAssessor** to the confidence-gating system.

> **Goal:** add a `RelevanceAssessor` that scores whether a step’s answer/tool-result is **relevant to the step task** (not just plausible).

In `my-react-agent`, confidence gating works like this:

1. After a step runs, the agent creates **step summary evidence** (a short factual summary).
2. `ConfidenceAssessor.assess_step_summary(...)` runs **all registered ParameterAssessors** on:
   - `query_text` = step task
   - `answer_text` = step summary content
3. It aggregates the per-assessor `ParameterRating.score` values into one confidence score.
4. If confidence is below threshold, the agent triggers a recovery loop (tries different actions/tools).

So, to add an assessor you must:

- Create a new class that extends `ParameterAssessor`
- Provide a `PromptId` + default `PromptTemplate` (so users don’t need to edit the framework)
- Return a `ParameterRating(name, score, reason, meta)`
- Register it (either directly, via factory list, or via plugin)

---

```python
from __future__ import annotations

"""
What this file does:
1) Define a new prompt_id string (we use a string constant for the demo).
2) Define a PromptTemplate for that id.
3) Implement RelevanceAssessor(ParameterAssessor) using that prompt.
4) Build ReActAgent with parameter_assessors=[RelevanceAssessor(...)].
5) Run and see confidence gating include assessor.

Run:
    python main_custom_assessor_demo.py

Try:
    Ask something that tempts the model to drift:
      "Tell me about Apple, but focus only on iPhones."
      "What's the capital of Germany? (Answer ONLY with the city.)"
"""

import importlib
import json
import logging
import re
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Any

import ollama

from my_react_agent.llm_adapters.llm_base import LLMBase

from my_react_agent.agent_prompts.prompt_registry import PromptRegistry
from my_react_agent.agent_prompts.prompt_template import PromptTemplate
from my_react_agent.agent_prompts.defaults_prompts import DEFAULT_PROMPTS

from my_react_agent.agent_core.agent_actions import (
    AnswerByItselfAction,
    ClarifyAction,
    UseToolAction,
    StopAction,
)
from my_react_agent.agent_core.agent_actions.need_context_action import NeedContextAction

from my_react_agent.agent_heart.react_agent import ReActAgent, ParameterAssessorFactory
from my_react_agent.agent_memory.llm_entity_extractor import LLMEntityExtractor

from my_react_agent.tool_management.tools.agent_tool import AgentTool
from my_react_agent.agent_memory.data_structures import Evidence

from my_react_agent.confidence_assessment.parameter_assessor import ParameterAssessor
from my_react_agent.confidence_assessment.models import ParameterRating


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 1) LLM adapter (Ollama -> LLMBase)
class OllamaGemmaLLM(LLMBase):
    def __init__(self, model: str = "gemma3:4b", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs) -> str:
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": self.temperature,
                "num_ctx": kwargs.get("num_ctx", 2048),
                "stop": kwargs.get("stop", ["<|endoftext|>"]),
            },
        )
        return response["response"]


# 2) Minimal tool (agent requires at least one tool)
class EchoTool(AgentTool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes back the input text (demo tool)."

    def execute(self, input: str) -> Evidence:
        return Evidence(
            tool=self.name,
            content=f"ECHO: {input}",
            url=None,
            extracted={"echo": input},
            as_of=datetime.utcnow(),
            confidence=0.9,
        )


# 3) Custom assessor prompt_id + default PromptTemplate (no core edits)
# In core, you'd add PromptId.CONF_RELEVANCE. For a one-file demo we just use a string.
CONF_RELEVANCE_PROMPT_ID = "confidence_relevance"

CONF_RELEVANCE_TEMPLATE = PromptTemplate(
    text=(
        "You are an evaluator for a QA system.\n\n"
        "Task: Score how RELEVANT the ANSWER is to the QUESTION.\n"
        "Relevance means: it directly addresses the asked topic and does not drift.\n\n"
        "Scoring:\n"
        "- 1.0 = clearly relevant and directly addresses the question.\n"
        "- 0.5 = partially relevant; some matches but some drift.\n"
        "- 0.0 = irrelevant / wrong subject / does not address the question.\n\n"
        "Output rules (CRITICAL):\n"
        "- Output MUST be a SINGLE JSON object and NOTHING else.\n"
        "- Keys MUST be exactly: score, reason.\n"
        "- score MUST be one of: 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0\n"
        "- reason MUST be <= 20 words.\n"
        "- Do NOT include any extra keys.\n\n"
        "Schema example: {schema_example}\n\n"
        "{knowledge_cutoff_block}{result_timestamp_block}"
        "QUESTION:\n{question}\n\n"
        "ANSWER:\n{answer}\n\n"
        "JSON:"
    ),
    required_vars={
        "schema_example",
        "knowledge_cutoff_block",
        "result_timestamp_block",
        "question",
        "answer",
    },
    description="Assess relevance of an answer to the question",
    version="1",
)


# 4) Custom ParameterAssessor: RelevanceAssessor
_JSON_OBJ_RE = re.compile(r"\{.*\}", flags=re.S)


def _extract_json_object(raw: str) -> dict:
    if not raw:
        return {}
    m = _JSON_OBJ_RE.search(raw)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _coerce_score(v: Any, fallback: float) -> float:
    try:
        f = float(v)
    except Exception:
        return fallback
    # clamp to [0, 1]
    if f < 0.0:
        f = 0.0
    if f > 1.0:
        f = 1.0
    # snap to 1 decimal to match prompt contract (optional)
    return float(f"{f:.1f}")


class RelevanceAssessor(ParameterAssessor):
    """
    Minimal assessor:
    - name: key in ratings dict ("relevance")
    - prompt_id: a string (in core you'd use PromptId.CONF_RELEVANCE.value)
    - default_prompt_template(): ships its own prompt so PromptRegistry can register it
    - assess(): returns ParameterRating(name, score, reason, meta)
    """

    @property
    def name(self) -> str:
        return "relevance"

    @property
    def prompt_id(self) -> str:
        return CONF_RELEVANCE_PROMPT_ID

    def default_prompt_template(self) -> PromptTemplate:
        # This is what ParameterAssessor.__init__ will register into prompts if missing.
        return CONF_RELEVANCE_TEMPLATE

    def assess(
        self,
        *,
        query_text: str,
        answer_text: str,
        tool_result_text: str = "",
        knowledge_cutoff: Optional[str] = None,
        result_timestamp: Optional[str] = None,
    ) -> ParameterRating:
        schema = {"score": 0.0, "reason": "short"}

        prompt = self._render_prompt(
            schema_example=json.dumps(schema),
            knowledge_cutoff_block=(knowledge_cutoff or ""),
            result_timestamp_block=(result_timestamp or ""),
            question=query_text,
            answer=answer_text,
        )

        raw = (self.llm.generate(prompt) or "").strip()
        obj = _extract_json_object(raw)

        score = _coerce_score(obj.get("score", self.default_fallback), self.default_fallback)
        reason = str(obj.get("reason", "fallback")).strip()[:120] or "fallback"

        return ParameterRating(
            name=self.name,
            score=score,
            reason=reason,
            meta={
                # optional flags used by ConfidenceAssessor._mean_score()
                # "exclude_from_mean": False,
            },
        )

# 5) Logging / prompts / optional factory assessors
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
    # Start from normal defaults
    prompts = PromptRegistry(_defaults=dict(DEFAULT_PROMPTS))

    # OPTIONAL BUT VERY CLEAR:
    # We also register the new prompt_id here so users see it explicitly.
    # (Even if you remove this, ParameterAssessor.__init__ will register it anyway.)
    prompts.register_default(CONF_RELEVANCE_PROMPT_ID, CONF_RELEVANCE_TEMPLATE)

    return prompts


def create_parameter_assessor_factories() -> List[ParameterAssessorFactory]:
    """
    Keep existing dynamic imports (optional). This demo focuses on the custom assessor.
    """
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


# 6) Main

def main() -> None:
    configure_logging(logging.INFO)

    prompts = create_prompt_registry()

    tools: Dict[str, object] = {"echo": EchoTool()}

    llms = {
        "planner_llm": OllamaGemmaLLM(),
        "summariser_llm": OllamaGemmaLLM(),
        "confidence_llm": OllamaGemmaLLM(),
        "refiner_llm": OllamaGemmaLLM(),
    }

    entity_extractor = LLMEntityExtractor(llms["summariser_llm"], prompts=prompts)

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

    # Register assessors:
    # - existing (optional)
    existing_factories = create_parameter_assessor_factories()

    # - custom (the point of this demo)
    custom_assessor = RelevanceAssessor(llms["confidence_llm"], prompts=prompts)

    # Mix factories + instances (ReActAgent.__init__ supports both)
    parameter_assessors: List[object] = []
    parameter_assessors.extend(existing_factories)
    parameter_assessors.append(custom_assessor)

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
        parameter_assessors=parameter_assessors,  # <-- includes RelevanceAssessor
    )

    print(
        "\nCustom Assessor Demo: RelevanceAssessor\n"
        f"- prompt_id: {CONF_RELEVANCE_PROMPT_ID}\n"
        "- assessor name: relevance\n\n"
        "Try:\n"
        "  What's the capital of Germany? (Answer ONLY with the city.)\n"
        "  Tell me about Apple, but focus only on iPhones.\n"
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
```

## Adding a Custom LLM Adapter (New `LLMBase` Implementation)

`my-react-agent` treats LLMs as **pluggable adapters**. Anything that implements the `LLMBase` interface can power the agent’s roles:

- **planner_llm** → creates step plans
- **summariser_llm** → synthesises step summaries + final answer
- **confidence_llm** → evaluates step quality/confidence (for retries)
- **refiner_llm** → turns (question + step task) into strict tool input

This section shows how to implement a new adapter and wire it into the agent, with an **Ollama DeepSeek** example.

---

#### Step 1 - The `LLMBase` contract

All adapters must implement:

```python
from __future__ import annotations

"""
What this shows:
1) Implement a new LLMBase adapter class (OllamaChatLLM) in ONE file
2) Use real Ollama models per role:
   - planner/summariser: deepseek-r1:8b
   - confidence/refiner: qwen2.5:3b
3) Run a tiny agent with one tool (CalculatorTool)

Prereqs:
- Ollama installed + running: https://ollama.com
- Pull models:
    ollama pull deepseek-r1:8b
    ollama pull qwen2.5:3b

Run:
    python main_custom_llm_adapter_demo.py

Try:
    What's the capital of Germany?
    Compute (23*7)-5
    2^10
"""

import ast
import importlib
import logging
import operator as op
import re
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional

import ollama

from my_react_agent.llm_adapters.llm_base import LLMBase
from my_react_agent.agent_prompts.prompt_registry import PromptRegistry
from my_react_agent.agent_prompts.defaults_prompts import DEFAULT_PROMPTS

from my_react_agent.agent_heart.react_agent import ReActAgent, ParameterAssessorFactory
from my_react_agent.agent_memory.llm_entity_extractor import LLMEntityExtractor

from my_react_agent.agent_core.agent_actions import (
    AnswerByItselfAction,
    ClarifyAction,
    UseToolAction,
    StopAction,
)
from my_react_agent.agent_core.agent_actions.need_context_action import NeedContextAction

from my_react_agent.tool_management.tools.agent_tool import AgentTool
from my_react_agent.agent_memory.data_structures import Evidence


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 1) NEW LLMBase adapter (this is the “custom adapter” you’re teaching)

class OllamaChatLLM(LLMBase):
    """
    Custom LLM adapter that satisfies LLMBase and talks to Ollama using the Chat API.

    Contract:
    - generate(prompt: str, **kwargs) -> str
    - accept kwargs (temperature/num_ctx/stop) so agent roles can override
    - raise a clear exception if Ollama is unreachable
    """

    def __init__(self, model: str, temperature: float = 0.1):
        self.model = model
        self.temperature = float(temperature)

    def generate(self, prompt: str, **kwargs) -> str:
        temp = float(kwargs.get("temperature", self.temperature))
        num_ctx = int(kwargs.get("num_ctx", 2048))
        stop = kwargs.get("stop", ["<|endoftext|>"])

        try:
            # Ollama chat endpoint: treat "prompt" as one user message
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temp,
                    "num_ctx": num_ctx,
                    "stop": stop,
                },
            )

            # ollama.chat returns an object/dict with message content
            msg = resp.get("message", {}) if isinstance(resp, dict) else getattr(resp, "message", None)
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            return (content or "").strip()

        except Exception as e:
            raise ConnectionError(
                f"Ollama error calling model={self.model!r}. "
                f"Is Ollama running on localhost:11434 and is the model pulled? "
                f"Original error: {e!r}"
            )


# 2) Minimal calculator tool (keep it simple; adapter demo is the point)
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
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BIN_OPS:
            return _ALLOWED_BIN_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARY_OPS:
            return _ALLOWED_UNARY_OPS[type(n.op)](_eval(n.operand))
        raise ValueError(f"Unsupported expression element: {type(n).__name__}")

    return float(_eval(node))


def _extract_math_expression(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"[0-9\.\s\+\-\*\/\(\)\%\^]+", text.strip())
    if not m:
        return None
    return (m.group(0).strip().replace("^", "**")) or None


class CalculatorTool(AgentTool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Safely evaluates basic arithmetic expressions."

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
        expr = _extract_math_expression(input or "")
        if not expr:
            return Evidence(
                tool=self.name,
                content="Calculator failed: no arithmetic expression found.",
                url=None,
                extracted={"error": True},
                as_of=datetime.utcnow(),
                confidence=0.2,
            )

        try:
            value = _safe_eval(expr)
            rendered = str(int(value)) if abs(value - round(value)) < 1e-12 else str(value)
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
                extracted={"error": True, "message": str(e), "expression": expr},
                as_of=datetime.utcnow(),
                confidence=0.1,
            )


# 3) Logging + prompts + optional assessor factories (unchanged patterns)

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
    """
    Same optional pattern you already use: build assessors if modules exist.
    If they don't, agent still runs (just less confidence gating).
    """
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


# 4) Main: wire custom adapter into the agent (this is the “how to use it” part)

def main() -> None:
    configure_logging(logging.INFO)
    prompts = create_prompt_registry()

    tools: Dict[str, object] = {"calculator": CalculatorTool()}

    # Mix models per role (common in practice)
    planner_llm = OllamaChatLLM(model="deepseek-r1:8b", temperature=0.2)
    summariser_llm = OllamaChatLLM(model="deepseek-r1:8b", temperature=0.1)
    confidence_llm = OllamaChatLLM(model="qwen2.5:3b", temperature=0.0)
    refiner_llm = OllamaChatLLM(model="qwen2.5:3b", temperature=0.0)

    entity_extractor = LLMEntityExtractor(summariser_llm, prompts=prompts)

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
        planner_llm=planner_llm,
        summariser_llm=summariser_llm,
        confidence_llm=confidence_llm,
        refiner_llm=refiner_llm,
        entity_extractor=entity_extractor,
        tools=tools,
        prompts=prompts,
        max_steps=8,
        step_actions=step_actions,
        low_conf_actions=low_conf_actions,
        parameter_assessors=parameter_assessors,
    )

    # show exactly what's configured at runtime
    used_assessors = []
    try:
        used_assessors = [
            type(a).__name__
            for a in (getattr(getattr(agent, "confidence_assessor", None), "parameter_assessors", []) or [])
        ]
    except Exception:
        used_assessors = []

    print(
        "\nCustom LLM Adapter Demo (LLMBase)\n"
        f"- planner/summariser: deepseek-r1:8b\n"
        f"- confidence/refiner: qwen2.5:3b\n"
        f"- confidence assessors: {used_assessors or ['(none loaded)']}\n"
        "\nTry:\n"
        "  What's the capital of Germany?\n"
        "  Compute (23*7)-5\n"
        "  2^10\n"
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
```

You can mix different models per role (common in practice):
- larger model for planning/summarisation
- cheaper/faster model for refinement/confidence