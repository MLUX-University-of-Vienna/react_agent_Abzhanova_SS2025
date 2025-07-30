from .entity_extractor import EntityExtractor
from llm_adapters.llm_base import LLMBase

class LLMEntityExtractor(EntityExtractor):
    def __init__(self, llm: LLMBase):
        self.llm = llm

    def extract(self, text: str) -> Optional[str]:
        prompt = f"""
Extract the most relevant named person or organisation from this sentence:
"{text}"

If none exists, return: NONE.
Only return the name itself.
"""
        result = self.llm.generate(prompt).strip()
        return None if result.upper() == "NONE" else result