import logging
from .llm_confidence_scorer import LLMConfidenceScorer
from .result_verifier_via_tool import ResultVerifierViaTool
from tool_management.tools.serpapi_tool import SerpAPITool
from llm_adapters.llm_base import LLMBase

logger = logging.getLogger(__name__)

class ConfidenceAssessor:
    def __init__(self, llm: LLMBase):
        self.llm_scorer = LLMConfidenceScorer(llm)
        self.verifier = ResultVerifierViaTool(
            llm,
            SerpAPITool(api_key="b53c73f7ff8666ff176932332e272be8175325eede05c4f1b716cbbccf6103ca")
        )
        self.llm = llm

    def assess(self, query: str, result: str, answer: str) -> float:
        llm_score = self.llm_scorer.score(query, result, answer)

        try:
            verification_question, verification_result = self.verifier.verify(query, answer)

            verifier_prompt = f"""Does the following information support the statement?

Statement: "{answer}"
Web Result:
{verification_result}

Answer YES or NO:"""

            response = self.llm.generate(verifier_prompt).strip().lower()
            verifier_score = 1.0 if "yes" in response else 0.3
        except Exception as e:
            logger.debug("Verification check failed, defaulting verifier score to 0.5", exc_info=e)
            verifier_score = 0.0

        final_score = round(0.5 * llm_score + 0.5 * verifier_score, 2)
        logger.debug(
            f"Confidence assessment summary:\n"
            f"- LLM Score: {llm_score}\n"
            f"- Verifier Score: {verifier_score}\n"
            f"- Final Combined Score: {final_score}"
        )
        return final_score