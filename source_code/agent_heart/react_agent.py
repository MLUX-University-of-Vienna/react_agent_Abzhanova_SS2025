import re
from typing import Dict, List
import logging

from llm_adapters.llm_base import LLMBase
from tool_management.tools.agent_tool import AgentTool
from tool_management.tool_selection.llm_based_strategy import LLMToolSelectionStrategy
from tool_management.tool_selection.regex_pattern_tool_selection_strategy import RegexPatternToolSelector
from query_separation.llm_based_query_separation import LLMQuerySeparationStrategy
from query_separation.logical_form_query_separation import LLMLogicalSeparationStrategy
from tool_management.tool_selection.hierarchical_tool_selection_strategy import HierarchicalToolSelectionStrategy
from task_management.task_manager import TaskManager

from agent_memory.conversation_memory import ConversationMemory
from agent_memory.query_memory import QueryMemory
from agent_memory.pronouns_resolution.pronouns_resolver import PronounsResolver
from agent_memory.pronouns_resolution.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

class ReActAgent:
    def __init__(self, query_separator_llm: LLMBase, tool_selector_llm: LLMBase,
                 answer_generator_llm: LLMBase, confidence_assessor_llm: LLMBase,
                 entity_extractor: EntityExtractor, pronouns_resolution_llm: LLMBase, tools: Dict[str, AgentTool]):
        self.separator = LLMLogicalSeparationStrategy(query_separator_llm)
        self.selector = RegexPatternToolSelector(tool_selector_llm)
        self.task_manager = TaskManager(tool_selector_llm, answer_generator_llm,
                                        confidence_assessor_llm, tools, self.selector)
        self.answer_generator_llm = answer_generator_llm

        self.conversation_memory = ConversationMemory(
            pronouns_resolver=PronounsResolver(entity_extractor, pronouns_resolution_llm)
        )
        self.query_memory = QueryMemory(
            pronouns_resolver=PronounsResolver(entity_extractor, pronouns_resolution_llm)
        )

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
        subqueries = self.separator.separate(query)
        logger.debug(f"Separated into {len(subqueries)} subquery(ies): {subqueries}")
        return subqueries
       
    def _answer_subqueries(self, subqueries: List[str]) -> List[Dict[str, str]]:
        results = []
        for subquery in subqueries:
            if self._contains_pronouns(subquery):
                resolved_subquery = self.query_memory.resolve_pronouns(subquery)
                if resolved_subquery == subquery:
                    resolved_subquery = self.conversation_memory.resolve_pronouns(subquery)
                    logger.debug("Pronouns resolved using conversation memory")
                else:
                    logger.debug("Pronouns resolved using query memory")
            else:
                resolved_subquery = subquery

            logger.info(f"Handling subquery: {resolved_subquery}")
            result = self.task_manager.run(resolved_subquery)

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

            self.query_memory.add_subquery(resolved_subquery, answer)
            self.conversation_memory.update(resolved_subquery, answer)

            results.append({"question": resolved_subquery, "answer": answer})
        return results

    def _aggregate_answers(self, original_query: str, parts: List[Dict[str, str]]) -> str:
        logger.info("Aggregating subquery answers into final response.")
        aggregating_prompt = "SYSTEM: You are an expert summarizer.\n\n"
        aggregating_prompt += "I asked these sub-questions and got these answers:\n\n"
        for p in parts:
            aggregating_prompt += f"- Q: {p['question']}\n  A: {p['answer']}\n\n"
        aggregating_prompt += (
            "Please combine these into a single, coherent final response to the original query:\n"
            f"\"{original_query}\"\n\nAnswer:"
        )
        return self.answer_generator_llm.generate(aggregating_prompt).strip()

    def _contains_pronouns(self, text: str) -> bool:
        pronouns = r'\b(he|she|it|they|his|her|their)\b'
        return re.search(pronouns, text, flags=re.IGNORECASE) is not None
