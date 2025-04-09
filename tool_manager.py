from abc import ABC, abstractmethod
from typing import dict

class ToolResponse:
    def __init__(self, content:str, success:bool = True, metadata: dict = None):
        self.content = content
        self.success = success
        self.metadata = metadata or {}
 
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
    def execute(self, input:str) -> ToolResponse:
        """Execute the tool with the given input"""
        pass

