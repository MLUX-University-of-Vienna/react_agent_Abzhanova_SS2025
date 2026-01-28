# my-react-agent

A **ReAct (Reason + Act)** agent framework for Python with **step-by-step traceability**, **evidence-first answering**, and **confidence-gated retries**.  
It plans a multi-step solution, executes each step via actions/tools, evaluates quality, and produces a final answer **grounded in collected observations and evidence**.

---

## Why this project

Most “agent” demos blur reasoning, tool calls, and final output into one opaque blob. `my-react-agent` is built for:

- **Auditability**: every step has an action, tool input/output, observation, evidence, and confidence.
- **Reliability**: low-confidence steps trigger a controlled recovery loop (retry with different actions/tools).
- **Extensibility**: add new tools/actions/plugins without rewriting the orchestration.

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
