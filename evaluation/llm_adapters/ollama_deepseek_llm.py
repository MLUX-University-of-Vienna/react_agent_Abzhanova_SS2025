import ollama
from .llm_base import LLMBase

class OllamaDeepseekLLM(LLMBase):
    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        temperature: float = 0.1,
        knowledge_cutoff: str = "2024",
        num_ctx: int = 2048,
        stop = ("\nObservation:", "<|endoftext|>"),
    ):
        self.model = model
        self.temperature = temperature
        self.knowledge_cutoff = knowledge_cutoff
        self.num_ctx = num_ctx
        self.stop = list(stop)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_ctx": kwargs.get("num_ctx", self.num_ctx),
                    "stop": kwargs.get("stop", self.stop),
                }
            )
            return response["response"]
        except Exception as e:
            raise ConnectionError(f"Ollama error: {str(e)}")
