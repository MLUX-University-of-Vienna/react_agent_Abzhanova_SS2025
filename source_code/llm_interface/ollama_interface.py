from .llm_interface import LLMInterface
import ollama

class OllamaInterface(LLMInterface):
    def __init__(self, model: str):
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        response = ollama.generate(
            model = self.model,
            prompt = prompt,
            options = kwargs
        )
        return response['response']