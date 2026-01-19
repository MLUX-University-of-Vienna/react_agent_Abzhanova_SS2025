# my-react-agent
A ReAct (Reason + Act) agent with explicit traceability, confidence gating, memory/evidence and pluggable tools.

This project implements a ReAct-style agent that decomposes a user question into step plans, executes each step via actions/tools, evaluates step quality and then synthesises a final answer only from collected observations and evidence.

## Key features:
- **Plan → Execute → Finalise pipeline** with step-by-step traceability (transcript + evidence per step)
- **Modular actions + handlers** (add new behaviour without touching core orchestration)
- **Pluggable tools** via a single execution boundary (`ToolExecutor`)
- **Memory + evidence-first design** (`QueryMemory` + `ConversationMemory`, structured `Evidence`)
- **Robustness hooks**: per-step confidence assessment + retry loops

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

## Usage
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
