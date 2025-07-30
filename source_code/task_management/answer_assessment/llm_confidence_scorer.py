import logging
from llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class LLMConfidenceScorer:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def score(self, query: str, result: str, answer: str) -> float:
        prompt = f'''You are a confidence grading assistant.

Evaluate the confidence (0.0 to 1.0) in the correctness of the given answer.

Question: {query}

Tool Result:
{result}

Answer:
{answer}

Scoring Criteria:
- Source reliability
- Information consistency
- Completeness
- Recency of information (the tool might be outdated)

Respond ONLY with a number between 0.0 and 1.0 (e.g., 0.85). Do NOT include any text or explanation.
'''
        try:
            response = self.llm.generate(prompt).strip()
            logger.debug(f"Raw LLM confidence output: '{response}'")
            return float(response)
        except Exception as e:
            logger.debug("Confidence assessment via LLM failed, defaulting to 0.5", exc_info=e)
            return 0.5
