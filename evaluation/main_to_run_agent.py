from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Optional, Tuple

from my_react_agent.agent_heart.react_agent import ReActAgent
from my_react_agent.agent_heart.action_handler_base import ActionHandler, empty_tool_call

from my_react_agent.tool_management.tools.agent_tool import AgentTool
from my_react_agent.agent_memory.data_structures import Evidence, Step, StepResult, StepToolCall, step_set_result

from my_react_agent.agent_core.agent_actions.action import Action
from my_react_agent.agent_core.agent_actions import (
    AnswerByItselfAction,
    ClarifyAction,
    UseToolAction,
    StopAction,
)
from my_react_agent.agent_core.agent_actions.need_context_action import NeedContextAction


class StarWarsAnswerAction(Action):
    @property
    def name(self) -> str:
        # Must match the handler registration key below
        return "STARWARS_ANSWER"

    @property
    def default_when_to_pick(self) -> str:
        return "Pick when the user's question is about Star Wars (characters, movies, lore, etc.)."

    @property
    def default_instructions(self) -> str:
        return (
            "Answer directly from general knowledge (no tools). "
            "The final answer MUST start with: 'May the Force be with you!' "
            "Then provide the answer."
        )

    @property
    def examples(self) -> list[str]:
        return [
            "User: Who is Luke Skywalker?",
            "User: Explain the Jedi and Sith.",
            "User: What is the Force in Star Wars?",
        ]


class StarWarsAnswerHandler(ActionHandler):
    @property
    def action_name(self) -> str:
        return "STARWARS_ANSWER"

    def run(self, ctx) -> Tuple[StepToolCall, StepResult, Step]:
        q = (ctx.question or "").strip()

        # Generate the Star Wars answer (LLM-only)
        prompt = (
            "You answer questions about Star Wars using general knowledge.\n"
            "Rules:\n"
            "- Be accurate; if uncertain, say so briefly.\n"
            "- Be concise unless the user asked for detail.\n\n"
            f"Question: {q}\n"
            "Answer:"
        )

        try:
            body = (ctx.agent.summariser_llm.generate(prompt) or "").strip()
        except Exception:
            body = ""

        if not body:
            body = "I’m not sure, but I can try to help if you rephrase the question."

        final = "May the Force be with you!\n\n" + body

        # Make the transcript show the final answer even though we stop early
        try:
            ctx.transcript_lines.append(f"Final answer: {final}")
        except Exception:
            pass

        ev = Evidence(
            tool="starwars_answer",
            content=final,
            url=None,
            extracted={"matched": True},
            as_of=datetime.utcnow(),
            confidence=0.9,
        )
        try:
            if getattr(ctx.step, "evidence", None) is not None:
                ctx.step.evidence.append(ev)
        except Exception:
            pass

        sr = StepResult(
            observation=final,
            final_answer=final,
            should_stop=True,   # <-- stop immediately once we produce the Star Wars answer
            success=True,
        )
        updated = step_set_result(ctx.step, sr)
        return empty_tool_call(tool=""), sr, updated

    def should_assess_result(self, ctx, *, step: Step, decision, step_result: StepResult) -> bool:
        # Deterministic routing; skip confidence gating for this.
        return False


def main() -> None:
    # Add Multiplication tool from the QuickStart)
    tools = [MultiplicationTool()]

    # Put custom action with defaults actions
    step_actions = [
        StarWarsAnswerAction(),
        NeedContextAction(),
        AnswerByItselfAction(),
        ClarifyAction(),
        UseToolAction(),
    ]

    # Also add custom action to low_conf_actions- otherwise low-confidence re-decisions will use the default list (which won’t include your custom action)
    low_conf_actions = [
        StarWarsAnswerAction(),
        NeedContextAction(),
        AnswerByItselfAction(),
        ClarifyAction(),
        UseToolAction(),
        StopAction(),
    ]

    action_handlers = {
        "STARWARS_ANSWER": StarWarsAnswerHandler(),
    }

    agent = ReActAgent.create(
        llm="llama3:8b",              
        tools=tools,
        step_actions=step_actions,
        low_conf_actions=low_conf_actions,
        action_handlers=action_handlers,
    )

    print("\nMultiplier tool + StarWarsAnswerAction\nType 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        print(agent.handle(user_input))
        print()


if __name__ == "__main__":
    main()
