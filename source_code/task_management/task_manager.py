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

    def assess(self, query: str, result: str, answer: str) -> float:
        prompt = f'''Evaluate the confidence (0-1) in this answer's accuracy:

Question: {query}
Tool Result: {result}
Generated Answer:{answer}

Consider:
1. Source reliability
2. Information consistency
3. Completeness
4. Remember that you can have outdated information

Respond with ONLY a number (0.0–1.0):'''
        try:
            llm_score = float(self.llm.generate(llm_prompt).strip())
        except Exception as e:
            logger.debug("Confidence assessment via LLM failed, defaulting to 0.5", exc_info=e)
            llm_score = 0.5

        if verifier_result:
            verifier_prompt = f"""Does the following information support the statement?

Statement: "{answer}"
Web Result:
{verifier_result}

Answer YES or NO."""   
            try:
                response = self.llm.generate(verifier_prompt).strip().lower()
                verifier_score = 1.0 if "yes" in response else 0.3
            except Exception as e:
                logger.debug("Verification check failed, defaulting verifier score to 0.5", exc_info=e)
                verifier_score = 0.5
        else:
            verifier_score = 0.5

        final_score = round(0.5 * llm_score + 0.5 * verifier_score, 2)
        logger.debug(
            f"Confidence assessment summary:\n"
            f"- LLM Score: {llm_score}\n"
            f"- Verifier Score: {verifier_score}\n"
            f"- Final Combined Score: {final_score}"
        )
        return final_score


class ResultVerifierViaTool:
    def __init__(self, llm: LLMInterface, verifier_tool: AgentTool):
        self.llm = llm
        self.verifier_tool = verifier_tool         

    def verify(self, original_query: str, answer: str) -> Tuple[str, str]:
        prompt = f"""Rephrase the original question into a yes/no question that checks the correctness of this answer.

Original Question: {original_query}
Answer: {answer}

Verification Question:"""

        verification_question = self.llm.generate(prompt).strip()
        logger.debug(f"Verification question:\n{verification_question}")

        result = self.verifier_tool.execute(verification_question)
        logger.debug(f"Verification result:\n{result.content}")

        return verification_question, result.content

class TaskManager:
    def __init__(
        self,
        tool_selector_llm: LLMInterface,
        answer_generator_llm: LLMInterface,
        confidence_assessor_llm: LLMInterface,
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