from abc import ABC, abstractmethod

class LLMBase(ABC):

    knowledge_cutoff: str = "UNKNOWN"

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass