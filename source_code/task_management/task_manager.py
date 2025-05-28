import re
import logging
from typing import Dict, Optional, Tuple, List
from llm.llm_interface import LLMInterface
from tool_management.tools.agent_tool import AgentTool, ToolResponse
from tool_management.tool_selection.tool_selection_strategy import ToolSelectionStrategy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 

class InitialThoughtGenerator:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def generate(self, query: str) -> str:
        prompt = f'''Analyze this question and decide if it requires external tools:

Question: {query}

Response format:
THOUGHT: <analysis>
NEEDS_TOOL: <yes|no>'''
        return self.llm.generate(prompt).strip()

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
        return self.llm.generate(prompt).strip()

    def generate_answer(self, query: str, result: str) -> str:
        prompt = f'''Based on this result, answer the query:

Query: {query}
Result: {result}

Format:
<think>analysis</think>
Final Answer: <answer>'''
        return self.llm.generate(prompt).strip()

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
            return float(self.llm.generate(prompt).strip())
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

        initial = self.generator.generate(query)
        thoughts.append(f"Initial Thought: {initial}")

        while True:
            # Checking confidence
            if final_answer and self.conf_assess.assess(final_answer) >= self.confidence_threshold:
                break

            # Determining available tools
            available = [t for t, count in tool_attempts.items() if count < self.max_attempts_per_tool]
            if not available and not final_answer:
                final_answer = self.ans_gen.fallback(query)
                break

            # Selecting tool via strategy
            candidates = self.selector.gather_candidates(query, self.tools)
            tool_name = self.selector.choose_tool(query, candidates, self.tools)
            if not tool_name:
                clarification = self.selector.ask_clarification(query)
                return {"clarification": clarification}

            # Executing and evaluatng
            tool_attempts[tool_name] += 1
            result = self.executor.execute(tool_name, query)
            actions.append(f"Used {tool_name}")
            observations.append(result)
            used_tools.append(tool_name)

            eval_ = self.evaluator.evaluate(query, result)
            thoughts.append(f"Evaluation: {eval_}")

            answer = self.evaluator.generate_answer(query, result)
            if not final_answer or self.conf_assess.assess(answer) > self.conf_assess.assess(final_answer):
                final_answer = answer

            if self.conf_assess.assess(answer) >= self.confidence_threshold:
                break

            if self.stuck_det.is_stuck(thoughts, actions):
                thoughts.append("Detected diminishing returns - terminating")
                break

        if not final_answer:
            final_answer = self.ans_gen.fallback(query)

        return {
            "final_answer": final_answer,
            "thought_process": "\n".join(thoughts),
            "actions_taken": "\n".join(actions),
            "observations": "\n".join(observations),
            "used_tools": used_tools
        }
