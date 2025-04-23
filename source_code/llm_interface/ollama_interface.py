import abc from ABC, abstractmethod
import ollama

class OllamaInterface(LLMInterface):
    def __init__(self, odel: str):
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        response = ollama.generate(
            model = self.model,
            prompt = prompt,
            options = kwargs
        )
        return response['response']