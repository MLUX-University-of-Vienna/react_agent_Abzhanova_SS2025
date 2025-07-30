import logging
from tool_management.tools.serpapi_tool import SerpAPITool
from tool_management.tools.agent_tool import AgentTool
from .result_verifier_via_tool import ResultVerifierViaTool
from llm_adapters.llm_base import LLMBase
logger = logging.getLogger(__name__)

class VerificationScorer:
    def __init__(self, llm: LLMBase, verifier_tool: AgentTool):
        tool = verifier_tool or SerpAPITool(api_key="b53c73f7ff8666ff176932332e272be8175325eede05c4f1b716cbbccf6103ca")
        self.verifier = ResultVerifierViaTool(llm, tool)
        self.llm = llm

    def score(self, query: str, answer: str) -> float:
        try:
            _, verification_result = self.verifier.verify(query, answer)

            prompt = f"""Does the following information support the statement?

Statement: "{answer}"
Web Result:
{verification_result}

Answer YES or NO:"""

            response = self.llm.generate(prompt).strip().lower()
            return 1.0 if "yes" in response else 0.3
        except Exception as e:
            logger.debug("Verification check failed, defaulting verifier score to 0.5", exc_info=e)
            return 0.5
