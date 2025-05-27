import re
import math
import logging
import ast 
from typing import Dict, Any, Optional
from .agent_tool import AgentTool, ToolResponse

class CalculatorTool(AgentTool):
    @property
    def name(self):
        return "calculator"
    
    @property
    def description(self): 
        return """Performs mathematical calculations. Supports:
        - Basic operations: +, -, *, /
        - Exponentiation: ^ or **
        - Factorials: !
        - Square roots: sqrt()
        - Percentages: %
        - Parentheses: ()"""

    _math_patterns = [
        r'calculat(e|ing|ion)|comput(e|ing)|solve|^how (much|many)',
        r'[\+\-\*/%^\(\)]|\d+\s*[\+\-\*/]\s*\d+|\d+!|!\d+|sqrt\(|\^|\*\*',
        r'what (is|are) (the )?(result|answer|value) (of|for)'
    ]

    _examples = """User: What is (15*3 + 2)/5?
{
  "thought": "I'll calculate this step-by-step",
  "action": "calculator",
  "action_input": "(15*3 + 2)/5",
  "final_answer": ""
}"""

    def _preprocess_expression(self, expr: str) -> str:
        try:
            expr = re.sub(r'(\d+)%\s*of\s*(\d+)', r'(\1/100)*\2', expr)
            expr = re.sub(r'(\d+)%', r'(\1/100)', expr)
            expr = expr.replace('^', '**')
            expr = re.sub(r'sqrt\(([^)]+)\)', r'math.sqrt(\1)', expr)
            expr = re.sub(r'(\d+)!', r'math.factorial(\1)', expr)
            cleaned_expr = re.sub(r'[^a-zA-Z0-9+\-*/(). sqrt!]', '', expr)
            logging.debug(f"Preprocessed expression: {cleaned_expr}")
            return cleaned_expr
        except Exception as e:
            logging.error(f"Expression preprocessing failed: {str(e)}")
            raise ValueError(f"Expression preprocessing failed: {str(e)}")

    def _is_safe_expression(self, expr: str) -> bool:
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
            ast.USub, ast.UAdd, ast.Load, ast.Call, ast.Attribute, ast.Name
        )
        try:
            tree = ast.parse(expr, mode='eval')
            for node in ast.walk(tree):
                if not isinstance(node, allowed_nodes):
                    if isinstance(node, ast.Attribute):
                        if node.attr in ['sqrt', 'factorial']:
                            continue
                    logging.warning(f"Disallowed node detected: {node}")
                    return False
                if isinstance(node, ast.Attribute):
                    if not hasattr(math, node.attr):
                        logging.warning(f"Unauthorized attribute access detected: {node.attr}")
                        return False
            return True
        except Exception:
            logging.error("AST parsing failed, unsafe expression detected.")
            return False

    def execute(self, input_expr: str) -> ToolResponse:
        try:
            logging.info(f"Original expression: {input_expr}")
            
            if not input_expr or not input_expr.strip():
                logging.warning("Received empty expression.")
                return ToolResponse(content="Error: Empty expression", success=False)
            
            if len(input_expr) > 100:
                logging.warning("Expression exceeds maximum allowed length.")
                return ToolResponse(content="Error: Expression too long", success=False)
            
            processed_expr = self._preprocess_expression(input_expr)
            logging.info(f"Processed expression: {processed_expr}")
            
            if not self._is_safe_expression(processed_expr):
                raise ValueError("Invalid or unsafe operations detected")
            
            safe_globals = {'math': math}
            result = eval(processed_expr, {'__builtins__': None}, safe_globals)
            
            if isinstance(result, float):
                result = round(result, 10)
                if result.is_integer():
                    result = int(result)
            
            logging.info(f"Calculation result: {result}")
            return ToolResponse(content=str(result), success=True)
            
        except Exception as e:
            logging.warning(f"Calculation failed for '{input_expr}': {str(e)}")
            return ToolResponse(content=f"Error: {str(e)}", success=False)

    @classmethod
    def get_patterns(cls):
        return cls._math_patterns
    
    @classmethod
    def get_examples(cls):
        return cls._examples
