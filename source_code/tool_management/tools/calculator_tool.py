import re
import math
import logging
import ast
from typing import Dict, Any, Optional, Union

from .agent_tool import AgentTool, ToolResponse

class CalculatorTool(AgentTool):
    MAX_LENGTH = 100
    SAFE_FUNCTIONS = {"sqrt", "factorial"}
    ALLOWED_AST_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.USub, ast.UAdd, ast.Load, ast.Call, ast.Attribute, ast.Name
    )

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return """Performs mathematical calculations. Supports:
        - Basic operations: +, -, *, /
        - Exponentiation: ^ or **
        - Factorials: !
        - Square roots: sqrt()
        - Percentages: %
        - Parentheses: ()"""

    _math_patterns = [
        r'\b(calculate|calculating|calculation|compute|computing|solve|evaluate|work out|figure out|simplify)\b',
        r'\bwhat (is|are|would be|could be|is the|is the result of)\b.*\b(sum|difference|product|quotient|value|answer|result|total)\b',
        r'\bhow (much|many|long|far|fast|tall)\b.*[\d+\-*/%^]',
        r'[-+*/%^=()]+',
        r'\b\d+\s*[-+*/^%]\s*\d+\b',
        r'\d+!?|\b!?\d+\b',
        r'\bsqrt\(|√\d+',
        r'\b\d+\s*(\^|\*\*)\s*\d+\b'
    ]

    _examples = """User: What is (15*3 + 2)/5?
{
  "thought": "I'll calculate this step-by-step",
  "action": "calculator",
  "action_input": "(15*3 + 2)/5",
  "final_answer": ""
}"""

    def execute(self, input_expr: str) -> ToolResponse:
        logging.info(f"Original expression: {input_expr}")

        validation_error = self._validate_expression(input_expr)
        if validation_error:
            return ToolResponse(content=validation_error, success=False)

        try:
            processed_expr = self._preprocess_expression(input_expr)
            logging.info(f"Processed expression: {processed_expr}")

            if not self._is_safe_expression(processed_expr):
                raise ValueError("Invalid or unsafe operations detected")

            result = self._evaluate_expression(processed_expr)
            logging.info(f"Calculation result: {result}")
            return ToolResponse(content=str(result), success=True)

        except Exception as e:
            logging.warning(f"Calculation failed for '{input_expr}': {str(e)}")
            return ToolResponse(content=f"Error: {str(e)}", success=False)

    def _validate_expression(self, expr: str) -> Optional[str]:
        expr = expr.strip()
        if not expr:
            return "Error: Empty expression"
        if len(expr) > self.MAX_LENGTH:
            return "Error: Expression too long"
        return None

    def _preprocess_expression(self, expr: str) -> str:
        try:
            expr = expr.lower().strip().rstrip("?.")

            expr = re.sub(r'multiply\s+(\d+)\s+(by|with)\s+(\d+)', r'\1*\3', expr)
            expr = re.sub(r'add\s+(\d+)\s+(and|to)\s+(\d+)', r'\1+\3', expr)
            expr = re.sub(r'subtract\s+(\d+)\s+from\s+(\d+)', r'\2-\1', expr)
            expr = re.sub(r'divide\s+(\d+)\s+by\s+(\d+)', r'\1/\2', expr)

            expr = re.sub(r'(\d+)%\s*of\s*(\d+)', r'(\1/100)*\2', expr)
            expr = re.sub(r'(\d+)%', r'(\1/100)', expr)
            expr = expr.replace('^', '**')
            expr = re.sub(r'sqrt\(([^)]+)\)', r'math.sqrt(\1)', expr)
            expr = re.sub(r'(\d+)!', r'math.factorial(\1)', expr)

            cleaned_expr = re.sub(r'[^a-zA-Z0-9+\-*/(). _]', '', expr)
            return cleaned_expr

        except Exception as e:
            logging.error(f"Expression preprocessing failed: {str(e)}")
            raise ValueError(f"Expression preprocessing failed: {str(e)}")


    def _is_safe_expression(self, expr: str) -> bool:
        try:
            tree = ast.parse(expr, mode='eval')
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    if node.attr not in self.SAFE_FUNCTIONS:
                        logging.warning(f"Unauthorized function: {node.attr}")
                        return False
                elif not isinstance(node, self.ALLOWED_AST_NODES):
                    logging.warning(f"Disallowed node detected: {type(node).__name__}")
                    return False
            return True
        except Exception:
            logging.error("AST parsing failed, unsafe expression detected.")
            return False

    def _evaluate_expression(self, expr: str) -> Union[int, float]:
        safe_globals = {'math': math}
        result = eval(expr, {'__builtins__': None}, safe_globals)

        if isinstance(result, float):
            result = round(result, 10)
            if result.is_integer():
                result = int(result)
        return result

    @classmethod
    def get_patterns(cls):
        return cls._math_patterns

    @classmethod
    def get_examples(cls):
        return cls._examples
