import re
from abc import ABC, abstractmethod
from typing import Dict,Any, Optional
import ast

class ToolResponse:
    def __init__(self, content:str, success:bool, metadata: Optional[Dict[str,Any]] = None):
        self.content = content
        self.success = success
        """metadata may contain contextual info/URLs/Source info/Debug Data"""
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

class CalculatorTool(AgentTool):
    @property
    def name(self):
        return  "Calculator"

    @property 
    def description(self): 
        return "Used for mathematical calculatins and numerical expressions"

    def evaluate_math_expression(self, expression: str) -> Optional[float]:
        """Safely evaluate a math expression using ast. Only allows basic arithmetic"""
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp,
            ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
            ast.USub, ast.UAdd, ast.Load, ast.Mod, ast.FloorDiv
        )
        
        node = ast.parse(expression, mode='eval')
        if not all(isinstance(n, allowed_nodes) for n in ast.walk(node)):
            raise ValueError("Invalid characters or operations used")
        
        return eval(compile(node, "<string>", "eval"))

    def execute(self, input: str) -> ToolResponse:
        try:
            sanitised = re.sub(r'[^0-9+\-*/(). ]', '', input)
            result = self.evaluate_math_expression(sanitised)
            return ToolResponse(content = str(result), success = True)
        except Exception as e:
            logging.warning(f"CalculatorTool error on input '{input}': {e}")
            return ToolResponse(content = f"Error: {str(e)}", success = False)