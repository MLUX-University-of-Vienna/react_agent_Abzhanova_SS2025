import re
import logging
from typing import Dict, Optional, Tuple, List
from llm_adapters.llm_base import LLMBase
from tool_management.tools.agent_tool import AgentTool, ToolResponse
from tool_management.tool_selection.tool_selection_strategy import ToolSelectionStrategy
from tool_management.tools.serpapi_tool import SerpAPITool

from .tool_executor import ToolExecutor
from .answer_generator import AnswerGenerator

from task_management.answer_assessment.llm_confidence_scorer import LLMConfidenceScorer
from task_management.answer_assessment.result_verifier_via_tool import ResultVerifierViaTool
from task_management.answer_assessment.confidence_assessor import ConfidenceAssessor
from task_management.answer_assessment.verification_scorer import VerificationScorer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

class TaskManager:
    def __init__(
        self,
        tool_selector_llm: LLMBase,
        answer_generator_llm: LLMBase,
        confidence_assessor_llm: LLMBase,
        tools: Dict[str, AgentTool],
        selector: ToolSelectionStrategy,
        confidence_threshold: float = 0.80,
        max_attempts_per_tool: int = 1
    ):
        self.logger = logger
        self.tool_selector_llm = tool_selector_llm
        self.answer_generator_llm = answer_generator_llm
        self.confidence_assessor_llm = confidence_assessor_llm
        self.tools = tools
        self.selector = selector
        self.confidence_threshold = confidence_threshold
        self.max_attempts_per_tool = max_attempts_per_tool

        self.tool_executor = ToolExecutor(tools)
        self.answer_generator = AnswerGenerator(answer_generator_llm)

        self.confidence_assessor = ConfidenceAssessor(confidence_assessor_llm)


    def run(self, query: str) -> Dict[str, object]:
        self.logger.info(f"Starting process for query: {query}")

        tool_attempts = {name: 0 for name in self.tools}
        tool_result_cache = {}
        best_answer = None
        best_confidence = 0.0

        while best_confidence < self.confidence_threshold:
            available_tools = self._get_available_tools(tool_attempts)
            self.logger.debug(f"Available tools: {available_tools}")
            
            if not available_tools:
                self.logger.info("No tools left, using fallback answer")
                return {"final_answer": self.answer_generator.fallback(query)}

            tool_name = self._select_tool(query, available_tools)
            if not tool_name:
                self.logger.info("Tool selection failed. Asking for clarification.")
                return {"clarification": self.selector.ask_clarification(query)}

            self.logger.info(f"Selected tool: {tool_name}")
            tool_attempts[tool_name] += 1
            result = self._get_tool_result(tool_name, query, tool_result_cache)
            self.logger.debug(f"Tool result: {result}")

            answer, confidence = self._evaluate_result(query, result)
            self.logger.debug(f"Generated answer: {answer}")
            self.logger.info(f"Confidence: {confidence:.2f}")

            if confidence == 0.0:
                self.logger.warning("Low confidence, skipping this result")
                continue
            
            if confidence > best_confidence:
                best_answer = answer
                best_confidence = confidence
                self.logger.info(f"Updated best answer with confidence {confidence:.2f}")

            if confidence >= self.confidence_threshold:
                self.logger.info("Confidence threshold met, stopping early")
                break

        if not best_answer:
            self.logger.warning("No confident answer found. Using fallback.")
            best_answer = self.answer_generator.fallback(query)

        return {"final_answer": self._extract_final_answer(best_answer)}
    
    def _get_available_tools(self, tool_attempts):
        return [t for t, attempts in tool_attempts.items() if attempts < self.max_attempts_per_tool]

    def _select_tool(self, query: str, available_tools: List[str]):
        available_tool_dict = {name: self.tools[name] for name in available_tools}
        candidates = self.selector.gather_candidates(query, available_tool_dict)
        return self.selector.choose_tool(query, candidates, available_tool_dict)


    def _get_tool_result(self, tool_name: str, query: str, cache: dict):
        if (tool_name, query) in cache:
            return cache[(tool_name, query)]
        result = self.tool_executor.execute(tool_name, query)
        cache[(tool_name, query)] = result
        return result

    def _evaluate_result(self, query: str, result: str):
        answer = self.answer_generator.generate(query, result)
        confidence = self.confidence_assessor.assess(query, result, answer)
        return answer, confidence

    def _extract_final_answer(self, answer: str) -> str:
        if "<answer>" in answer:
            return answer.split("<answer>")[-1].strip().rstrip("</answer>")
        return answer.strip()