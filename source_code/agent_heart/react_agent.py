import re
from typing import Dict, List
import logging

from llm.llm_interface import LLMInterface
from tool_management.tools.agent_tool import AgentTool
from tool_management.tool_selection.llm_based_strategy import LLMToolSelectionStrategy
from tool_management.tool_selection.regex_pattern_tool_selection_strategy import RegexPatternToolSelector
from query_separation.llm_based_query_separation import LLMQuerySeparationStrategy
from query_separation.logical_form_query_separation import LLMLogicalSeparationStrategy
from tool_management.tool_selection.hierarchical_tool_selection_strategy import HierarchicalToolSelectionStrategy
from task_management.task_manager import TaskManager

from agent_memory.conversation_memory import ConversationMemory
from agent_memory.query_memory import QueryMemory

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

class ReActAgent:
    def __init__(self, query_separator_llm: LLMInterface, tool_selector_llm: LLMInterface,
                 answer_generator_llm: LLMInterface, confidence_assessor_llm: LLMInterface,
                 tools: Dict[str, AgentTool]):
        self.separator = LLMLogicalSeparationStrategy(query_separator_llm)
        self.selector = RegexPatternToolSelector(tool_selector_llm)
        self.task_manager = TaskManager(tool_selector_llm, answer_generator_llm,
                                        confidence_assessor_llm, tools, self.selector)
        self.answer_generator_llm = answer_generator_llm

        self.conversation_memory = ConversationMemory()
        self.query_memory = QueryMemory()

    def handle(self, query: str) -> str:
        logger.info(f"Received query: {query}")
        self.query_memory.start_new(query) 

        subqueries = self._separate_query(query)
        results = self._answer_subqueries(subqueries)
        final_answer = self._aggregate_answers(query, results)

        logger.info(f"Final aggregated answer: {final_answer}")
        self.conversation_memory.update(query, final_answer)
        return final_answer

    def _separate_query(self, query: str) -> List[str]:
        subqs = self.separator.separate(query)
        logger.debug(f"Separated into {len(subqs)} subquery(ies): {subqs}")
        return subqs
       
    def _answer_subqueries(self, subqueries: List[str]) -> List[Dict[str, str]]:
        results = []
        for sq in subqueries:
            if self._contains_pronouns(sq):
                resolved_sq = self.query_memory.resolve_pronouns(sq)
                if resolved_sq == sq:
                    resolved_sq = self.conversation_memory.resolve_pronouns(sq)
                    logger.debug("Pronouns resolved using conversation memory")
                else:
                    logger.debug("Pronouns resolved using query memory")
            else:
                resolved_sq = sq

            logger.info(f"Handling subquery: {resolved_sq}")
            result = self.task_manager.run(resolved_sq)

            if "final_answer" in result:
                raw = result["final_answer"]
                answer = raw.replace("Final Answer:", "").strip()
                logger.debug(f"Subquery result: {answer}")
            elif "clarification" in result:
                answer = f"[Needs clarification] {result['clarification']}"
                logger.warning(f"Subquery clarification required: {answer}")
            else:
                answer = "[No answer produced]"
                logger.error("Unexpected result structure from task manager")

            self.query_memory.add_subquery(resolved_sq, answer)
            self.conversation_memory.update(resolved_sq, answer)

            results.append({"question": resolved_sq, "answer": answer})
        return results

    def _aggregate_answers(self, original_query: str, parts: List[Dict[str, str]]) -> str:
        logger.info("Aggregating subquery answers into final response.")
        agg_prompt = "SYSTEM: You are an expert summarizer.\n\n"
        agg_prompt += "I asked these sub-questions and got these answers:\n\n"
        for p in parts:
            agg_prompt += f"- Q: {p['question']}\n  A: {p['answer']}\n\n"
        agg_prompt += (
            "Please combine these into a single, coherent final response to the original query:\n"
            f"\"{original_query}\"\n\nAnswer:"
        )
        return self.answer_generator_llm.generate(agg_prompt).strip()

    def _contains_pronouns(self, text: str) -> bool:
        pronouns = r'\b(he|she|it|they|his|her|their)\b'
        return re.search(pronouns, text, flags=re.IGNORECASE) is not None
