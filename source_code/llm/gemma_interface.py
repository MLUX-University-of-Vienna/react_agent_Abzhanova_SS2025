import ollama
from llm.llm_interface import LLMInterface

class GemmaInterface(LLMInterface):
    def __init__(self, model: str = "gemma3:4b", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature