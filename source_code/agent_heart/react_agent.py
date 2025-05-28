from typing import Dict
from llm.llm_interface import LLMInterface
from tool_management.tools.agent_tool import AgentTool
from tool_management.tool_selection.llm_based_strategy import LLMToolSelectionStrategy
from tool_selection.regex_pattern_tool_selection_strategy import RegexPatternToolSelector
from query_separation.llm_based_query_separation import LLMQuerySeparationStrategy
from tool_management.tool_selection.hierarchical_tool_selection_strategy import HierarchicalToolSelectionStrategy
from task_management.task_manager import TaskManager

class ReActAgent:
    def __init__(self, llm: LLMInterface, tools: Dict[str, AgentTool]):
        self.separator = LLMQuerySeparationStrategy(llm)
        self.selector  = RegexPatternToolSelector(llm)
        self.task_manager = TaskManager(llm, tools, self.selector)
        self.llm = llm

    def handle(self, query: str) -> str:
        # 1) It's separating into subqueries
        subqs = self.separator.separate(query)
        # 2) Answering each in turn
        parts = []
        for sq in subqs:
            result = self.task_manager.run(sq)
            parts.append({"question": sq, "answer": result["final_answer"]})

        agg_prompt = "SYSTEM: You are an expert summarizer.\n\n"
        agg_prompt += "I asked these sub-questions and got these answers:\n\n"
        for p in parts:
            agg_prompt += f"- Q: {p['question']}\n  A: {p['answer']}\n\n"
        agg_prompt += (
            "Please combine these into a single, coherent final response to the original query:\n"
            f"\"{query}\"\n\nAnswer:"
        )

        final = self.llm.generate(agg_prompt)
        return final.strip()