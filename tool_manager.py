from abc import ABC, abstractmethod

class AgentTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the tool"""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
         """A human-readable description of the tool's purpose"""
        pass
        
    @abstractmethod
    def execute(self, input: str) -> str:
        """Execute the tool with the given input"""
        pass

