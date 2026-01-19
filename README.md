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

pip install my-react-agent
