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
        if "Final Answer:" not in answer:
            return 0.0
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
        llm: LLMInterface,
        tools: Dict[str, AgentTool],
        selector: ToolSelectionStrategy,
        confidence_threshold: float = 0.85,
        max_attempts_per_tool: int = 2
    ):
        self.logger = logger
        self.llm = llm
        self.tools = tools
        self.selector = selector
        self.confidence_threshold = confidence_threshold
        self.max_attempts_per_tool = max_attempts_per_tool

        self.tool_executor = ToolExecutor(tools)
        self.answer_generator = AnswerGenerator(llm)
        self.confidence_assessor = ConfidenceAssessor(llm)

    def run(self, query: str) -> Dict[str, object]:
        self.logger.info(f"Starting process for query: {query}")

        tool_attempts = {name: 0 for name in self.tools}
        tool_result_cache = {}
        best_answer = None
        best_confidence = 0.0

        while True:
            available_tools = self._get_available_tools(tool_attempts)
            if not available_tools:
                self.logger.info("No tools left, using fallback answer")
                return {"final_answer": self.answer_generator.fallback(query)}

            tool_name = self._select_tool(query)
            if not tool_name:
                return {"clarification": self.selector.ask_clarification(query)}

            tool_attempts[tool_name] += 1
            result = self._get_tool_result(tool_name, query, tool_result_cache)

            if (tool_name, query) in tool_result_cache:
                result = tool_result_cache[(tool_name, query)]
            else:
                result = self.executor.execute(tool_name, query)
                tool_result_cache[(tool_name, query)] = result

            # Executing and evaluatng
            actions.append(f"Used {tool_name}")
            observations.append(result)
            used_tools.append(tool_name)

            eval_ = self.evaluator.evaluate(query, result)
            thoughts.append(f"Evaluation: {eval_}")

            answer = self.evaluator.generate_answer(query, result)
            answer_conf = self.conf_assess.assess(answer)

            if previous_answer and answer.strip().lower() == previous_answer.strip().lower():
                thoughts.append("Answer converged — same as previous, stopping")
                break
            previous_answer = answer

            if not final_answer or answer_conf > final_confidence:
                final_answer = answer
                final_confidence = answer_conf
                thoughts.append(f"Promoted better answer with confidence {answer_conf:.2f}")
            
            if answer_conf >= acceptable_confidence:
                thoughts.append("High confidence (>= 0.90) — exiting immediately")
                break
            elif answer_conf >= self.confidence_threshold:
                thoughts.append(f"Confidence ({answer_conf:.2f}) above threshold — exiting")
                break
            
            if self.stuck_det.is_stuck(thoughts, actions):
                thoughts.append("Detected diminishing returns - terminating")
                break

        if not final_answer:
            final_answer = self.ans_gen.fallback(query)
            thoughts.append("Fallback triggered after all logic paths")

        logger.info(f"\n=== Final LLM Output ===\n{final_answer}")
        logger.info(f"\n=== Thought Process ===\n{chr(10).join(thoughts)}")

        return {
            "final_answer": final_answer,
            "thought_process": "\n".join(thoughts),
            "actions_taken": "\n".join(actions),
            "observations": "\n".join(observations),
            "used_tools": used_tools
        }
    
    def _get_available_tools(self, tool_attempts):
        return [t for t, attempts in tool_attempts.items() if attempts < self.max_attempts_per_tool]

    def _select_tool(self, query: str):
        candidates = self.selector.gather_candidates(query, self.tools)
        return self.selector.choose_tool(query, candidates, self.tools)

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