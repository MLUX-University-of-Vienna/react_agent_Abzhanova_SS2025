import ollama
from .llm_base import LLMBase

class OllamaGemmaLLM(LLMBase):
    def __init__(self, model: str = "gemma3:4b", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_ctx": kwargs.get("num_ctx", 2048),
                    "stop": kwargs.get("stop", ["<|endoftext|>"])
                }
            )
            return response['response']
        except Exception as e:
            raise ConnectionError(f"Ollama error: {str(e)}")