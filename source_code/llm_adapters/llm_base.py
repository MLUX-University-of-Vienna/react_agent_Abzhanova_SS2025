from abc import ABC, abstractmethod

class LLMBase(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass