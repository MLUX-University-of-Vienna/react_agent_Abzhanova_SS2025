import re
import logging
from typing import Dict, Optional, Tuple, List
from llm.llm_interface import LLMInterface
from tool_management.tools.agent_tool import AgentTool, ToolResponse
from tool_management.tool_selection.tool_selection_strategy import ToolSelectionStrategy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

class ToolExecutor:
    def __init__(self, tools: Dict[str, AgentTool]):
        self.tools = tools

    def execute(self, name: str, query: str) -> str:
        if name in self.tools:
            result = self.tools[name].execute(query)
            return result.content if result.success else f"Tool error: {result.content}"
        return "Invalid tool"


class AnswerGenerator:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def generate(self, query: str, result: str) -> str:
        prompt = f"""Based on this result, answer the query:

Query: {query}
Result: {result}

Format:
Final Answer: <your answer here>"""
        return self.llm.generate(prompt).strip()

    def fallback(self, query: str) -> str:
        prompt = f"""Unable to get a confident answer via tools.
Please answer this directly:

Question: {query}

Final Answer:"""
        return self.llm.generate(prompt).strip()

class ConfidenceAssessor:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def assess(self, answer: str) -> float:
        prompt = f'''Rate the confidence (0-1) in this answer's accuracy:

{answer}

Consider:
1. Source reliability
2. Information consistency
3. Completeness

Respond only with the confidence score (0.0-1.0):'''
        try:
            output = self.llm.generate(prompt).strip()
            logger.debug(f"ConfidenceAssessor Output:\n{output}")
            return float(output)
        except:
            return 0.5

class TaskManager:
    def __init__(
        self,
        tool_selector_llm: LLMInterface,
        answer_generator_llm: LLMInterface,
        confidence_assessor_llm: LLMInterface,
        tools: Dict[str, AgentTool],
        selector: ToolSelectionStrategy,
        confidence_threshold: float = 0.80,
        max_attempts_per_tool: int = 2
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
        confidence = self.confidence_assessor.assess(answer)
        return answer, confidence

    def _extract_final_answer(self, answer: str) -> str:
        if "<answer>" in answer:
            return answer.split("<answer>")[-1].strip().rstrip("</answer>")
        return answer.strip()