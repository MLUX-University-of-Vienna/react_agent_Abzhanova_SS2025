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

class InitialThoughtGenerator:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def generate(self, query: str) -> str:
        prompt = f'''Analyze this question and decide if it requires external tools:

Question: {query}

Response format:
THOUGHT: <analysis>
NEEDS_TOOL: <yes|no>'''
        output = self.llm.generate(prompt).strip()
        logger.debug(f"InitialThoughtGenerator Output:\n{output}")
        return output

class ToolExecutor:
    def __init__(self, tools: Dict[str, AgentTool]):
        self.tools = tools

    def execute(self, name: str, query: str) -> str:
        if name in self.tools:
            result = self.tools[name].execute(query)
            return result.content if result.success else f"Tool error: {result.content}"
        return "Invalid tool"

class ResultEvaluator:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def evaluate(self, query: str, result: str) -> str:
        prompt = f'''Evaluate this tool result for the query:

Query: {query}
Result: {result}

Assessment:''' 
        output = self.llm.generate(prompt).strip()
        logger.debug(f"ResultEvaluator.evaluate Output:\n{output}")
        return output

    def generate_answer(self, query: str, result: str) -> str:
        prompt = f'''Based on this result, answer the query:

Query: {query}
Result: {result}

Format:
<think>analysis</think>
Final Answer: <answer>'''
        output = self.llm.generate(prompt).strip()
        logger.debug(f"ResultEvaluator.generate_answer Output:\n{output}")
        return output

class AnswerGenerator:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def direct(self, query: str) -> str:
        prompt = f'''Answer this question concisely:

Question: {query}

Answer:''' 
        return self.llm.generate(prompt).strip()

    def fallback(self, query: str) -> str:
        prompt = f'''After careful consideration:

Question: {query}

Response:''' 
        output = self.llm.generate(prompt).strip()
        logger.debug(f"AnswerGenerator.direct Output:\n{output}")
        return output

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

class StuckDetector:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def measure_similarity(self, t1: str, t2: str) -> float:
        w1, w2 = set(t1.lower().split()), set(t2.lower().split())
        inter = w1 & w2
        union = w1 | w2
        return len(inter) / len(union) if union else 0.0

    def is_stuck(self, thoughts: List[str], actions: List[str]) -> bool:
        if len(actions) < 2:
            return False
        last_actions = [a.split(':')[0] for a in actions[-2:]]
        if len(set(last_actions)) == 1:
            return True
        last_thoughts = thoughts[-2:]
        if len(last_thoughts[0]) > 50 and len(last_thoughts[1]) > 50:
            return self.measure_similarity(last_thoughts[0], last_thoughts[1]) > 0.8
        return False

class TaskManager:
    def __init__(
        self,
        llm: LLMInterface,
        tools: Dict[str, AgentTool],
        selector: ToolSelectionStrategy,
        confidence_threshold: float = 0.85,
        max_attempts_per_tool: int = 2
    ):
        self.logger = logging.getLogger(__name__)
        self.generator = InitialThoughtGenerator(llm)
        self.executor = ToolExecutor(tools)
        self.evaluator = ResultEvaluator(llm)
        self.ans_gen = AnswerGenerator(llm)
        self.conf_assess = ConfidenceAssessor(llm)
        self.stuck_det = StuckDetector(llm)
        self.tools = tools
        self.selector = selector
        self.confidence_threshold = confidence_threshold
        self.max_attempts_per_tool = max_attempts_per_tool

    def run(self, query: str) -> Dict[str, object]:
        self.logger.info(f"Starting process for query: {query}")
        thoughts, actions, observations, used_tools = [], [], [], []
        tool_attempts = {t: 0 for t in self.tools.keys()}
        final_answer = None
        final_confidence = 0.0
        previous_answer = None
        tool_result_cache = {}

        initial = self.generator.generate(query)
        thoughts.append(f"Initial Thought: {initial}")

        while True:
            # Checking confidence
            if final_answer and final_confidence >= self.confidence_threshold:
                thoughts.append(f"Confidence threshold ({final_confidence:.2f}) met — terminating")
                break

            # Determining available tools
            available = [t for t, count in tool_attempts.items() if count < self.max_attempts_per_tool]
            if not available and not final_answer:
                final_answer = self.ans_gen.fallback(query)
                thoughts.append("No tools left, using fallback answer")
                break

            # Selecting tool via strategy
            candidates = self.selector.gather_candidates(query, self.tools)
            tool_name = self.selector.choose_tool(query, candidates, self.tools)
            if not tool_name:
                clarification = self.selector.ask_clarification(query)
                return {"clarification": clarification}

            # Caching of tool results
            tool_attempts[tool_name] += 1
            if (tool_name, query) in tool_result_cache:
                result = tool_result_cache[(tool_name, query)]
            else:
                result = self.executor.execute(tool_name, query)
                tool_result_cache[(tool_name, query)] = result

            # Executing and evaluatng
            result = self.executor.execute(tool_name, query)
            actions.append(f"Used {tool_name}")
            observations.append(result)
            used_tools.append(tool_name)

            eval_ = self.evaluator.evaluate(query, result)
            thoughts.append(f"Evaluation: {eval_}")

            answer = self.evaluator.generate_answer(query, result)
            answer_conf = self.conf_assess.assess(answer)

            if not final_answer or answer_conf > final_confidence:
                final_answer = answer
                final_confidence = answer_conf
                thoughts.append(f"Promoted better answer with confidence {answer_conf:.2f}")
            
            if answer_conf >= self.confidence_threshold:
                thoughts.append("Answer confidence threshold met - terminating")
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
