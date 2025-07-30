import logging
from typing import Tuple
from llm.llm_interface import LLMInterface
from tool_management.tools.agent_tool import AgentTool

logger = logging.getLogger(__name__)

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
