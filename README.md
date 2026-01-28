# my-react-agent

A **ReAct (Reason + Act)** agent framework for Python with **step-by-step traceability**, **evidence-first answering**, and **confidence-gated retries**.  
It plans a multi-step solution, executes each step via actions/tools, evaluates quality, and produces a final answer **grounded in collected observations and evidence**.

---

## Why this project

LangChain and LlamaIndex are strong frameworks—but they’re optimised for *different priorities*:

- **LangChain** is an **integration + composition** system (chains, agents, tool wrappers, retrievers, many providers). It’s great when you want to assemble an app quickly from lots of building blocks.
- **LlamaIndex** is a **data/RAG framework** (ingestion, indexing, retrieval, routing, structured querying). It’s great when your core problem is “connect LLMs to your data” at scale.

`my-react-agent` exists for a different goal: **a small, inspectable agent runtime where traceability, evidence and reliability are first-class — not optional add-ons**.

### What you get here that’s harder to guarantee in LangChain/LlamaIndex

- **Traceability as a core invariant (not a plugin / external service dependency)**  
  Every step *must* produce a structured record: action decision → tool input/output → observation → evidence → confidence.  
  This makes debugging and evaluation predictable because the “paper trail” is built into the runtime.

- **Evidence-first answering as a default design**  
  The final answer is synthesized from collected **observations + `Evidence` objects**, making it straightforward to enforce “don’t invent facts” policies and to display citations/snippets in a consistent format.

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
  Creates a step plan, runs each step deterministically, then synthesizes a final answer.
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
  Centralized prompt management (`PromptRegistry`) with overridable defaults.
- **Plugin support**  
  Optional runtime extension via `REACT_AGENT_PLUGINS`.

---

### From PyPI
pip install my-react-agent

## License
MIT

## Requirements
- Python 3.10+
- Ollama (local LLM runtime)

## From PIP
pip install my-react-agent

## From Source
pip install git+https://git01lab.cs.univie.ac.at/zhaniyaa77/my-react-agent.git

## Install Ollama
#### Download and install Ollama:
- https://ollama.com/download

#### Pull a model (example used below: llama3):
ollama pull llama3

## Quickstart
```python
import os

from my_react_agent.agent_heart.react_agent import ReActAgent
from my_react_agent.llm_adapters.ollama_llama3_llm import OllamaLlama3LLM

from my_react_agent.agent_core.agent_actions import (
    AnswerByItselfAction,
    ClarifyAction,
    UseToolAction,
    StopAction,
)
from my_react_agent.agent_core.agent_actions.need_context_action import NeedContextAction

from my_react_agent.agent_memory.llm_entity_extractor import LLMEntityExtractor


def main() -> None:
    # LLM roles (all backed by Ollama)
    planner_llm = OllamaLlama3LLM(model="llama3")
    summariser_llm = OllamaLlama3LLM(model="llama3")
    confidence_llm = OllamaLlama3LLM(model="llama3")

    # Entity extractor used by the NEED_CONTEXT mechanism
    entity_extractor = LLMEntityExtractor(summariser_llm)

    # Minimal tool set: empty dict works if you don't use tools
    # If your package includes tools and you want them, you can create them here.
    tools = {}

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
        planner_llm=planner_llm,
        summariser_llm=summariser_llm,
        confidence_llm=confidence_llm,
        entity_extractor=entity_extractor,
        tools=tools,
        max_steps=6,
        step_actions=step_actions,
        low_conf_actions=low_conf_actions,
    )

    answer = agent.handle("Explain what a ReAct agent is in 2 sentences.")
    print(answer)
if __name__ == "__main__":
    main()
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

**Prompts**
- `PromptRegistry`  
  Stores prompt templates for planning, refinement, confidence assessment, summarisation.

**Plugins**
- Loaded via `REACT_AGENT_PLUGINS` environment variable  
  A plugin module exposes `plugin.register(ctx)` and can add tools/actions/assessors/prompts.


## Adding a Custom Action
- Adding a Custom Action (Example: GreetingAction)

- Goal: If the user starts with a greeting (e.g., “hi”, “hello”, “good morning”), the agent should include a greeting back in the final answer.

- In my-react-agent, a custom action has two parts:
1. Action definition (Action) — metadata the LLM sees in the action catalogue
2. Action handler (ActionHandler) — runtime code executed when the action is selected

- How actions are selected and executed
At runtime the agent:
1. Builds an action catalogue from step_actions (and low_conf_actions during retries).
2. Lets the planner LLM choose one action name for the current step.
3. Maps action name → handler in ReActAgent._get_handler_for_action.
4. Executes the handler and records observation/evidence.

- So, to add a new action you must:
1. Create a new Action class (e.g., GreetingAction)
2. Create a new ActionHandler class (e.g., GreetingHandler)
3. Register the action in step_actions
4. Add a mapping in _get_handler_for_action

#### Step 1 - Create GreetingAction
```python
from __future__ import annotations

from my_react_agent.agent_core.agent_actions.action import Action


class GreetingAction(Action):
    @property
    def name(self) -> str:
        # Must match the handler map key in ReActAgent._get_handler_for_action
        return "GREETING"

    @property
    def default_when_to_pick(self) -> str:
        return (
            "Pick when the user message contains a greeting (hi/hello/hey/good morning/etc.) "
            "and we should greet back in the final answer."
        )

    @property
    def default_instructions(self) -> str:
        return (
            "Detect greeting intent in the user's message. "
            "If present, prepare a short friendly greeting to include in the final answer. "
            "Do not answer the main question here; just prepare the greeting."
        )

    @property
    def examples(self) -> list[str]:
        return [
            "User: Hi! What is the capital of Germany?",
            "User: Hello, can you explain ReAct agents?",
            "User: Good morning — what is the weather in Tokyo?",
        ]
```
- Notes:
1. name must be unique and stable.
2. The planner LLM uses when_to_pick and instructions to decide whether to select this action.

#### Step 2 - Create GreetingHandler
- Create: my_react_agent/agent_heart/react_handlers/greeting.py
- This handler:
1. detects if the user question contains a greeting,
2. stores a greeting in the agent context (_context_snippets),
3. returns a small observation (traceable in transcript).

```python
from __future__ import annotations

import re
from datetime import datetime
from typing import Tuple, TYPE_CHECKING

from my_react_agent.agent_memory.data_structures import (
    Step,
    StepResult,
    StepToolCall,
    ToolResponse,
    Evidence,
    step_set_result,
)
from my_react_agent.agent_heart.action_handler_base import ActionHandler, empty_tool_call

if TYPE_CHECKING:
    from my_react_agent.agent_heart.action_context import ActionHandlerContext


_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening)\b",
    flags=re.I,
)


class GreetingHandler(ActionHandler):
    @property
    def action_name(self) -> str:
        return "GREETING"

    def run(self, ctx: "ActionHandlerContext") -> Tuple[StepToolCall, StepResult, Step]:
        user_text = (ctx.question or "").strip()

        greeting_text = ""
        if _GREETING_RE.search(user_text):
            greeting_text = "Hello!"
            # Stored for final synthesis
            ctx.agent._context_snippets.append(f"GREETING: {greeting_text}")

        observation = greeting_text or "No greeting detected."
        step_result = StepResult(
            observation=observation,
            final_answer=None,
            should_stop=False,
            success=True,
        )

        # Optional: attach evidence for traceability
        ev = Evidence(
            tool="greeting",
            content=observation,
            url=None,
            extracted={"greeting": greeting_text or "", "matched": bool(greeting_text)},
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

    def should_assess_result(
        self,
        ctx: "ActionHandlerContext",
        *,
        step: Step,
        decision,
        step_result: StepResult,
    ) -> bool:
        # Greeting detection is deterministic; no need to confidence-gate it.
        return False
```

#### Step 3 — Wire the handler into ReActAgent
Update ReActAgent._get_handler_for_action to include the new handler:

```python
from my_react_agent.agent_heart.react_handlers.greeting import GreetingHandler

def _get_handler_for_action(self, action_name: str) -> ActionHandler:
    handler_map = {
        "ANSWER_BY_ITSELF": AnswerByItselfHandler(),
        "STOP": StopHandler(),
        "CLARIFY": ClarifyHandler(),
        "RESOLVE_PRONOUNS_AND_OMITTED ENTITIES": NeedContextHandler(),
        "USE_TOOL": UseToolHandler(),
        "GREETING": GreetingHandler(),  
    }
    return handler_map.get(action_name, UseToolHandler())
```

#### Step 4 — Register the action in step_actions
When constructing your agent:
```python
from my_react_agent.agent_core.agent_actions.greeting_action import GreetingAction

step_actions = [
    GreetingAction(),   
    NeedContextAction(),
    AnswerByItselfAction(),
    ClarifyAction(),
    UseToolAction(),
    StopAction(),
]
```

## Adding a Custom Tool (Example: `PictureAnalyserTool`)

In `my-react-agent`, a **tool** is any component that implements the `AgentTool` interface:

- **Input**: a single `str` (`tool_input`)
- **Output**: an `Evidence` object (structured, traceable, timestamped)

Tools are executed through a single boundary (`ToolExecutor`) and are typically triggered by the `USE_TOOL` action (via `UseToolAction`).

This section shows how to add a new tool: a **picture analyser** that reads an image from disk and returns structured evidence.

---

#### Step 1 - Create the tool class

Create: `evaluation/tools/picture_analyser_tool.py`

```python
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional

from my_react_agent.agent_memory.data_structures import Evidence
from my_react_agent.tool_management.tools.agent_tool import AgentTool

try:
    from PIL import Image
except Exception:
    Image = None


class PictureAnalyserTool(AgentTool):
    _PATH_RE = re.compile(r"(?:^|\s)path:(?P<path>\S+)", flags=re.I)
    _QUESTION_RE = re.compile(r"(?:^|\s)question:(?P<q>.+)$", flags=re.I)

    @property
    def name(self) -> str:
        return "picture_analyser"

    @property
    def description(self) -> str:
        return (
            "Analyse an image from a local file path and return basic properties "
            "(size, format, mode) and simple heuristics. "
            "Input should include path:<file> and optionally question:<...>."
        )

    @property
    def refiner_instructions(self) -> str:
        return (
            "Return ONE line in the format:\n"
            "path:<file_path> question:<what to analyze>\n"
            "Rules:\n"
            "- Must include path:\n"
            "- Use the exact file path from the user message if present\n"
            "- Do NOT output JSON\n"
            "- Keep it under 200 characters if possible\n"
            "Examples:\n"
            "path:./img/cat.jpg question:Describe what you see\n"
            "path:/tmp/photo.png question:Read any visible text"
        )

    @property
    def refiner_input_format(self) -> str:
        return "path:<file_path> question:<what to analyse>"

    @property
    def refiner_input_regex(self) -> Optional[str]:
        # Simple validation: must contain "path:" and some non-space path.
        return r"^.*\bpath:\S+.*$"

    @property
    def refiner_forbidden(self) -> str:
        return "Forbidden: JSON, newlines, URLs instead of file paths."

    @property
    def refiner_max_chars(self) -> int:
        return 300

    # --- Core execution ---
    def execute(self, input: str) -> Evidence:
        if Image is None:
            return Evidence(
                tool=self.name,
                content="PictureAnalyserTool requires Pillow (PIL). Install: pip install pillow",
                url=None,
                extracted={"error": True, "reason": "pillow_missing"},
                as_of=datetime.utcnow(),
                confidence=0.1,
            )

        raw = (input or "").strip()
        path = self._extract_path(raw)
        question = self._extract_question(raw)

        if not path:
            return Evidence(
                tool=self.name,
                content="Missing image path. Provide: path:<file_path> question:<...>",
                url=None,
                extracted={"error": True, "reason": "missing_path", "tool_input": raw},
                as_of=datetime.utcnow(),
                confidence=0.1,
            )

        if not os.path.exists(path):
            return Evidence(
                tool=self.name,
                content=f"Image file not found: {path}",
                url=None,
                extracted={"error": True, "reason": "file_not_found", "path": path},
                as_of=datetime.utcnow(),
                confidence=0.1,
            )

        try:
            with Image.open(path) as img:
                w, h = img.size
                mode = img.mode
                fmt = (img.format or "").upper()

                # very lightweight "analysis"
                notes = []
                if w >= 2000 or h >= 2000:
                    notes.append("high_resolution")
                if mode in ("RGBA", "LA"):
                    notes.append("has_alpha")

                # Optional: compute a tiny heuristic (average brightness) without heavy ML
                avg_brightness = None
                try:
                    gray = img.convert("L")
                    small = gray.resize((64, 64))
                    px = list(small.getdata())
                    avg_brightness = sum(px) / max(1, len(px))  # 0..255
                except Exception:
                    pass

            content_lines = [
                f"PATH: {path}",
                f"FORMAT: {fmt or '(unknown)'}",
                f"SIZE: {w}x{h}",
                f"MODE: {mode}",
            ]
            if question:
                content_lines.append(f"QUESTION: {question}")
            if notes:
                content_lines.append(f"NOTES: {', '.join(notes)}")
            if avg_brightness is not None:
                content_lines.append(f"AVG_BRIGHTNESS: {avg_brightness:.1f}/255")

            return Evidence(
                tool=self.name,
                content="\n".join(content_lines),
                url=None,
                extracted={
                    "path": path,
                    "format": fmt,
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "notes": notes,
                    "avg_brightness": avg_brightness,
                    "question": question,
                },
                as_of=datetime.utcnow(),
                confidence=0.8,
            )

        except Exception as e:
            return Evidence(
                tool=self.name,
                content=f"Failed to analyse image: {e!r}",
                url=None,
                extracted={"error": True, "reason": "exception", "path": path},
                as_of=datetime.utcnow(),
                confidence=0.1,
            )

    def _extract_path(self, s: str) -> str:
        m = self._PATH_RE.search(s or "")
        if not m:
            return ""
        return (m.group("path") or "").strip().strip('"').strip("'")

    def _extract_question(self, s: str) -> str:
        m = self._QUESTION_RE.search(s or "")
        if not m:
            return ""
        return (m.group("q") or "").strip()
```

 What this tool does: 
- Takes a path:<file> input
- Loads the image using Pillow
- Returns an Evidence object with structured fields in extracted

#### Step 2 - Create the tool class
in evaluation/main_to_run_agent.py: 

```python
def _init_picture_analyser_tool() -> Optional[object]:
    try:
        from .tools.picture_analyser_tool import PictureAnalyserTool
        return PictureAnalyserTool()
    except Exception as e:
        logger.exception("[tools] picture_analyser failed to init: %r", e)
        return None


def create_tools() -> Dict[str, object]:
    tools: Dict[str, object] = {}

    # ... existing tools ...

    pa = _init_picture_analyser_tool()
    if pa is not None:
        tools["picture_analyser"] = pa

    logger.info("[create_tools] Tools initialised count=%d keys=%s", len(tools), list(tools.keys()))
    return tools
```

#### How the agent decides to call your tool
- The planner LLM sees each tool’s:
1. name
2. description
- When it chooses USE_TOOL, it outputs a tool_name and the refiner produces tool_input.

- To make your tool easier to select:
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

#### Step 1 — Add a new PromptId

Add a new ID to `my_react_agent/agent_prompts/prompts_ids.py`:

```python
class PromptId(str, Enum):

    # Confidence assessor (existing)
    CONF_ENTITY_ALIGNMENT = "confidence_entity_alignment"
    CONF_ANSWER_QUALITY = "confidence_answer_quality"
    CONF_ANSWER_REALISM = "confidence_answer_realism"

    # Add this:
    CONF_RELEVANCE = "confidence_relevance"
```

#### Step 2 - Add the default prompt template
- Add a default prompt template to my_react_agent/agent_prompts/defaults_prompts.py under DEFAULT_PROMPTS.
- IMPORTANT: your PromptTemplate.required_vars must match what your assessor passes to _render_prompt().

```python
from .prompts_ids import PromptId
from .prompt_template import PromptTemplate

DEFAULT_PROMPTS: dict[PromptId, PromptTemplate] = {
    # ... existing ...

    PromptId.CONF_RELEVANCE: PromptTemplate(
        text=(
            "You are an evaluator for a QA system.\n\n"
            "Task: Score how RELEVANT the ANSWER is to the QUESTION.\n"
            "Relevance means: it directly addresses the asked topic and does not drift to another entity or subject.\n\n"
            "Scoring:\n"
            "- 1.0 = clearly relevant and directly addresses the question.\n"
            "- 0.5 = partially relevant; some content matches but key parts drift or are generic.\n"
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
    ),
}
```

#### Step 3 — Implement RelevanceAssessor
-Create: my_react_agent/confidence_assessment/relevance_assessor.py

```python
from __future__ import annotations

import json
import logging
from typing import Optional

from ..llm_adapters.llm_base import LLMBase
from ..agent_prompts.defaults_prompts import DEFAULT_PROMPTS
from ..agent_prompts.prompt_template import PromptTemplate
from ..agent_prompts.prompts_ids import PromptId
from ..agent_prompts.prompt_registry import PromptRegistry

from .json_utils import _coerce_score_0_1, _extract_json_object, _pick_reason, _round2
from .models import ParameterRating
from .parameter_assessor import ParameterAssessor

logger = logging.getLogger(__name__)


class RelevanceAssessor(ParameterAssessor):
    def __init__(
        self,
        llm: LLMBase,
        *,
        prompts: PromptRegistry,
        default_fallback: float = 0.5,
        log_parse_failures: bool = True,
    ):
        super().__init__(llm, prompts=prompts, default_fallback=default_fallback, log_parse_failures=log_parse_failures)
        logger.info(
            "[RelevanceAssessor.__init__] ready prompt_id=%s fallback=%.2f",
            self.prompt_id,
            self.default_fallback,
        )

    @property
    def name(self) -> str:
        # This key becomes part of the ratings dict in ConfidenceAssessor
        return "relevance"

    @property
    def prompt_id(self) -> str:
        return PromptId.CONF_RELEVANCE.value

    def default_prompt_template(self) -> PromptTemplate:
        tpl = DEFAULT_PROMPTS.get(PromptId.CONF_RELEVANCE) or DEFAULT_PROMPTS.get(self.prompt_id)
        if tpl is None:
            raise KeyError(f"DEFAULT_PROMPTS missing template for {PromptId.CONF_RELEVANCE!r} / {self.prompt_id!r}")
        return tpl

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
            knowledge_cutoff_block=knowledge_cutoff or "",
            result_timestamp_block=result_timestamp or "",
            question=query_text,
            answer=answer_text,
        )

        score = self.default_fallback
        reason = "fallback"
        raw = ""

        try:
            raw = (self.llm.generate(prompt) or "").strip()
            obj = _extract_json_object(raw) or {}

            if not obj and self.log_parse_failures:
                logger.warning("[RelevanceAssessor] JSON parse failed raw=%r", raw[:600])

            score = _coerce_score_0_1(obj.get("score", score), score)
            reason = _pick_reason(obj, reason)

        except Exception as e:
            logger.warning("[RelevanceAssessor] LLM failed error=%r raw=%r", e, raw[:500])

        return ParameterRating(
            name=self.name,
            score=_round2(score),
            reason=reason,
            meta={
                # Optional: if you want to exclude from mean when irrelevant:
                # "exclude_from_mean": False,
            },
        )
```

-What this returns:
1. ParameterRating.name: "relevance" (unique key)
2. ParameterRating.score: float 0–1 (rounded)
3. ParameterRating.reason: short explanation

#### Step 4 — Register it (Factory method)
Add these in evaluation/main_to_run_agent.py:

```python
candidates = [
    ("my_react_agent.confidence_assessment.entity_alignment_assessor", "EntityAlignmentAssessor"),
    ("my_react_agent.confidence_assessment.answer_quality_assessor", "AnswerQualityAssessor"),
    ("my_react_agent.confidence_assessment.answer_realism_assessor", "AnswerRealismAssessor"),
    # Add this:
    ("my_react_agent.confidence_assessment.relevance_assessor", "RelevanceAssessor"),
]
```

- Because factory pattern already does:
```python
factories.append(lambda llm, prompts, _cls=cls: _cls(llm, prompts=prompts))
```
that is all what we need.

#### Step 3 — Implement RelevanceAssessor