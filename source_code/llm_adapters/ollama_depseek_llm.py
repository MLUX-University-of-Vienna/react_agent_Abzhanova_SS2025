import ollama
from llm.llm_interface import LLMInterface

class OllamaDeepseekLLM(LLMBase):
    def __init__(self, model: str = "deepseek-r1:8b", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_ctx": 2048,
                    "stop": ["\nObservation:", "<|endoftext|>"]
                }
            )
            return response['response']
        except Exception as e:
            raise ConnectionError(f"Ollama error: {str(e)}")