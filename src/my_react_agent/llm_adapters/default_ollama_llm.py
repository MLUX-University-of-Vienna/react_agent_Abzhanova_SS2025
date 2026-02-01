from .llm_base import LLMBase
from typing import Tuple

class DefaultOllamaLLM(LLMBase):
    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        num_ctx: int = 2048,
        stop: Tuple[str, ...] = ("<|endoftext|>",),
    ):
        self.model = model
        self.temperature = temperature
        self.num_ctx = int(num_ctx or 2048)
        self.stop = list(stop or ("<|endoftext|>",))

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            import ollama
        except Exception as e:
            raise ConnectionError(f"Ollama error: {str(e)}")

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_ctx": kwargs.get("num_ctx", self.num_ctx),
                    "stop": kwargs.get("stop", self.stop),
                },
            )
            return response["response"]
        except Exception as e:
            raise ConnectionError(f"Ollama error: {str(e)}")