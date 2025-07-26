from .entity_extractor import EntityExtractor
from llm.llm_interface import LLMInterface

class LLMEntityExtractor(EntityExtractor):
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def extract(self, text: str) -> Optional[str]:
        prompt = f"""
Extract the most relevant named person or organization from this sentence:
"{text}"

If none exists, return: NONE.
Only return the name itself.
"""
        result = self.llm.generate(prompt).strip()
        return None if result.upper() == "NONE" else result