import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ToolResponse:
    def __init__(self, content: str, success: bool, metadata: Optional[Dict[str, Any]] = None):
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
    def execute(self, input: str) -> ToolResponse:
        """Execute the tool with the given input"""
        pass

    def _contains_category(self, text: str, patterns: list) -> bool:
        """Checks if the text matches any of the regex patterns."""
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    @classmethod
    def get_examples(cls) -> str:
        """Return the examples specific to each tool"""
        raise NotImplementedError("Each tool must define its own examples.")
    
    @classmethod
    def get_patterns(cls) -> list:
        """Return the regex patterns specific to each tool"""
        raise NotImplementedError("Each tool must define its own patterns.")
