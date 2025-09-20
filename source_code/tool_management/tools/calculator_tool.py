# tool_management/tools/calculator_tool.py
import math
import logging
import ast
import re
from typing import Optional, Union, Dict, Set
from datetime import datetime

from .agent_tool import AgentTool
from agent_core.models import ToolResponse, Observation

logger = logging.getLogger(__name__)


class CalculatorTool(AgentTool):
    MAX_INPUT_LENGTH = 300
    SAFE_FUNCTIONS = {"sqrt", "factorial", "floor", "ceil", "log", "log10", "exp"}
    SAFE_ATTRIBUTES = {"pi", "e"}
    ALLOWED_AST_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Load,
        ast.Call,
        ast.Attribute,
        ast.Name,
    )

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluates a single safe Python math expression"

    @property
    def refiner_instructions(self) -> str:
        return (
            "Output ONE Python expression only — no prose.\n"
            "Allowed tokens:\n"
            "  • digits with optional decimal/scientific notation (e.g., 12, 3.5, 1e-3)\n"
            "  • parentheses ( ... )\n"
            "  • operators: +  -  *  /  //  %  **  (never ^)\n"
            "  • functions: round(expr, N) and math.sqrt|math.floor|math.ceil|math.log|math.log10|math.exp|math.factorial\n"
            "  • constants: math.pi, math.e\n"
            "  • explicit multiplication only (use *)\n"
            "Do not use attribute formatting like .format(...)."
        )

    @property
    def refiner_input_format(self) -> str:
        return (
            "Mini-grammar (informal):\n"
            "  <expr> := <term> ( [+-] <term> )*\n"
            "  <term> := <power> ( [*/%//] <power> )*\n"
            "  <power> := <atom> ( ** <atom> )*\n"
            "  <atom> := <number> | ( <expr> ) | round( <expr> , <int> ) | math.<fn>( <expr> ) | math.<const>\n"
            "  <number> := digits [ '.' digits ] [ e|E [ +|- ] digits ]\n"
            "  <fn> := sqrt | floor | ceil | log | log10 | exp | factorial\n"
            "  <const> := pi | e\n"
            "\n"
            "Examples (valid tool inputs):\n"
            "  (2000/330000000)*100\n"
            "  round((450/1200)*100, 2)\n"
            "  (15/100)*800\n"
            "  (12 + 3) * 5\n"
            "  math.floor(3.7)\n"
            "  23**2 * math.sqrt(4)\n"
            "  round(66663.432424234, 2)\n"
            "  2*math.pi*3\n"
        )

    @property
    def refiner_examples(self) -> str:
        return "\n".join(
            [
                "(2000/330000000)*100",
                "round((450/1200)*100, 2)",
                "(15/100)*800",
                "(12 + 3) * 5",
                "math.floor(3.7)",
                "23**2 * math.sqrt(4)",
                "(250/400)*100",
                "round(66663.432424234, 2)",
                "2*math.pi*3",
            ]
        )

    @property
    def refiner_input_regex(self) -> Optional[str]:
        return None

    @property
    def refiner_forbidden(self) -> str:
        return (
            "No attribute formatting like .format(...). No f-strings. No assignments or comparisons. "
            "No unknown names. No dunders. No imports. No comprehensions. No slicing."
        )

    @property
    def refiner_max_chars(self) -> int:
        return self.MAX_INPUT_LENGTH

    @classmethod
    def get_examples(cls) -> str:
        return "\n".join(
            [
                "round(66663.432424234, 2)",
                "(1 + 2) * 3",
                "(250/400)*100",
                "math.sqrt(81) + 7",
                "math.floor(3.7)",
                "2*math.pi*3",
            ]
        )

    def execute(self, input_string: str) -> ToolResponse:
        logger.info(f"[calculator] Received raw input length={len(input_string) if input_string is not None else 0}")
        preprocessed = self._preprocess_input(input_string)
        if preprocessed != input_string:
            logger.info(f"[calculator] Preprocessed input applied old={input_string!r} new={preprocessed!r}")
        else:
            logger.debug("[calculator] No preprocessing rewrite was necessary")
        validation_error = self._validate_input(preprocessed)
        if validation_error:
            logger.warning(f"[calculator] Validation failed message={validation_error!r}")
            return self._error(validation_error)
        try:
            if not self._is_safe_expression(preprocessed, allowed_names={"math", "round"}):
                logger.warning("[calculator] Safety gate rejected the expression")
                raise ValueError("Invalid or unsafe operations detected")
            logger.debug("[calculator] Safety gate accepted the expression, proceeding to AST parse")
            tree = ast.parse(preprocessed, mode="eval")
            typing_info = self._infer_result_type(tree)
            value = self._evaluate_expression(preprocessed)
            display_text = self._format_value(value)
            unit = None
            if typing_info.get("unit") == "percent" and typing_info.get("already_scaled", False):
                unit = "percent"
                display_text = f"{self._format_percent(value)}"
            observation = Observation(
                value=float(value) if isinstance(value, (int, float)) else None,
                unit=unit,
                raw_text=f"{preprocessed} = {display_text}",
                source="calculator",
                observed_at=datetime.utcnow(),
                meta={
                    "expr": preprocessed,
                    "expr_kind": typing_info.get("expr_kind"),
                    "result_unit": typing_info.get("unit"),
                    "already_scaled": typing_info.get("already_scaled", False),
                },
            )
            logger.info("[calculator] Evaluation succeeded and observation produced")
            return ToolResponse(success=True, observations=[observation], metadata={"tool": self.name})
        except Exception as e:
            logger.warning(f"[calculator] Execution failed error={e!r}")
            return self._error(f"Error: {e}", error=e)

    def _preprocess_input(self, input_string: str) -> str:
        s = (input_string or "").strip()
        if not s:
            logger.debug("[calculator._preprocess_input] Empty input after strip")
            return s
        s_one_space = re.sub(r"\s+", " ", s)
        if s_one_space != s:
            logger.debug("[calculator._preprocess_input] Normalised internal whitespace")
        s = s_one_space
        m = re.fullmatch(r"\(\s*(?P<inner>.+?)\s*\)\.format\(\s*(?P<digits>\d+)\s*\)", s, flags=re.IGNORECASE)
        if m:
            inner = m.group("inner")
            digits = m.group("digits")
            rewritten = f"round(({inner}), {digits})"
            logger.info("[calculator._preprocess_input] Rewrote .format(...) to round(..., N)")
            return rewritten
        if " or " in s.lower():
            parts = re.split(r"\bor\b", s, flags=re.IGNORECASE)
            first = (parts[0] or "").strip()
            logger.info(f"[calculator._preprocess_input] Detected boolean alternative, selecting first option candidate={first!r}")
            s = first
        s = s.strip()
        return s

    def _validate_input(self, input_string: str) -> Optional[str]:
        if not input_string or not input_string.strip():
            return "Error: Empty input"
        s = input_string.strip()
        if len(s) > self.MAX_INPUT_LENGTH:
            return "Error: Input too long"
        return None

    def _is_safe_expression(self, expression: str, allowed_names: Set[str]) -> bool:
        try:
            tree = ast.parse(expression, mode="eval")
            for node in ast.walk(tree):
                if not isinstance(node, self.ALLOWED_AST_NODES):
                    logger.debug(f"[calculator._is_safe_expression] Disallowed node type={type(node).__name__}")
                    return False
                if isinstance(node, ast.Tuple):
                    logger.debug("[calculator._is_safe_expression] Tuple detected")
                    return False
                if isinstance(node, ast.Name):
                    if node.id not in allowed_names:
                        logger.debug(f"[calculator._is_safe_expression] Disallowed name id={node.id!r}")
                        return False
                if isinstance(node, ast.Attribute):
                    if not isinstance(node.value, ast.Name) or node.value.id != "math":
                        logger.debug("[calculator._is_safe_expression] Attribute base not math")
                        return False
                    if isinstance(getattr(node, "ctx", None), ast.Load):
                        if node.attr not in self.SAFE_FUNCTIONS and node.attr not in self.SAFE_ATTRIBUTES:
                            logger.debug(f"[calculator._is_safe_expression] Disallowed math attribute attr={node.attr!r}")
                            return False
                if isinstance(node, ast.Call):
                    if node.keywords:
                        logger.debug("[calculator._is_safe_expression] Keyword args detected")
                        return False
                    if isinstance(node.func, ast.Name) and node.func.id == "round":
                        if len(node.args) != 2:
                            logger.debug("[calculator._is_safe_expression] round called with wrong arity")
                            return False
                        arg1 = node.args[1]
                        if not isinstance(arg1, ast.Constant) or not isinstance(arg1.value, int):
                            logger.debug("[calculator._is_safe_expression] round second argument is not an int constant")
                            return False
                    elif isinstance(node.func, ast.Attribute):
                        if node.func.attr not in self.SAFE_FUNCTIONS:
                            logger.debug(f"[calculator._is_safe_expression] Disallowed math function attr={node.func.attr!r}")
                            return False
                        if len(node.args) != 1:
                            logger.debug("[calculator._is_safe_expression] math.<fn> called with wrong arity")
                            return False
                    else:
                        logger.debug("[calculator._is_safe_expression] Call target not allowed")
                        return False
            return True
        except Exception as e:
            logger.debug(f"[calculator._is_safe_expression] AST parse/scan failed error={e!r}")
            return False

    def _evaluate_expression(self, expression: str) -> Union[int, float]:
        safe_globals = {"math": math, "round": round, "__builtins__": None}
        result = eval(expression, safe_globals, {})
        if isinstance(result, float):
            result = round(result, 12)
            if result.is_integer():
                result = int(result)
        return result

    def _format_value(self, value: Union[int, float]) -> str:
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            text = f"{value:.12f}".rstrip("0").rstrip(".")
            return text if text else "0"
        return str(value)

    def _format_percent(self, value: Union[int, float]) -> str:
        if isinstance(value, (int, float)):
            v = float(value)
            return f"{v:.2f}%"
        return str(value)

    def _error(self, message: str, error: Exception = None) -> ToolResponse:
        return ToolResponse(
            success=False,
            observations=[
                Observation(
                    raw_text=message,
                    source="calculator",
                    observed_at=datetime.utcnow(),
                )
            ],
            metadata={"tool": self.name, **({"error": repr(error)} if error else {})},
        )

    def _infer_result_type(self, tree: ast.AST) -> Dict[str, Union[str, bool]]:
        has_division = False
        has_multiplication = False
        has_multiply_by_100 = False
        has_divide_by_100 = False
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Mult):
                    has_multiplication = True
                    if self._is_const_100(node.left) or self._is_const_100(node.right):
                        has_multiply_by_100 = True
                elif isinstance(node.op, ast.Div):
                    has_division = True
                    if self._is_const_100(node.right):
                        has_divide_by_100 = True
        if has_division and has_multiply_by_100:
            return {"expr_kind": "percent_relative", "unit": "percent", "already_scaled": True}
        if has_multiplication and has_divide_by_100:
            return {"expr_kind": "percent_of", "unit": None, "already_scaled": False}
        if has_division and not has_multiply_by_100 and not has_divide_by_100:
            return {"expr_kind": "ratio", "unit": None, "already_scaled": False}
        return {"expr_kind": "arithmetic", "unit": None, "already_scaled": False}

    def _is_const_100(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            try:
                return float(node.value) == 100.0
            except Exception:
                return False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd) and isinstance(node.operand, ast.Constant):
            try:
                return float(node.operand.value) == 100.0
            except Exception:
                return False
        return False
