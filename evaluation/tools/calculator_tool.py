from __future__ import annotations

import ast
import logging
import math
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Dict, Optional, Set, Union

from my_react_agent.agent_memory.data_structures import Evidence
from my_react_agent.tool_management.tools.agent_tool import AgentTool

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

    ALLOWED_NAMES_DEFAULT = {"math", "round"}

    FLOAT_ROUND_DECIMALS = 12

    FLOAT_FORMAT_DECIMALS = 12

    PERCENT_FORMAT_DECIMALS = 2
    PERCENT_BASE_VALUE = 100.0

    ERROR_EMPTY_INPUT = "Error: Empty input"
    ERROR_INPUT_TOO_LONG = "Error: Input too long"
    ERROR_UNSAFE_EXPRESSION = "Invalid or unsafe operations detected"
    ERROR_AMBIGUOUS_COMMAS = (
        "Error: Ambiguous comma usage in number. Use plain digits or standard "
        "thousands separators."
    )

    EXPR_KIND_PERCENT_RELATIVE = "percent_relative"
    EXPR_KIND_PERCENT_OF = "percent_of"
    EXPR_KIND_RATIO = "ratio"
    EXPR_KIND_ARITHMETIC = "arithmetic"
    UNIT_PERCENT = "percent"

    _THOUSANDS_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")
    _ANY_DIGIT_COMMA_DIGIT_RE = re.compile(r"(?<=\d),(?=\d)")
    _THOUSANDS_COMMA_POS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Arithmetic-only evaluator, CANNOT be used if any number is given or for external data lookup."

    @property
    def refiner_instructions(self) -> str:
        return (
            "Output ONE Python expression only s— no prose.\n"
            "Allowed: numbers, parentheses, + - * / // % **, round(expr,N), math.pi, math.e.\n"
            "Always recheck parentheses: incorrect grouping changes the result.\n"
            "If a quantity is described verbally (e.g., million, billion, percent), "
            "convert it using exact powers of 10 ONLY. \n"
	    "Do not invent numbers, numbers used in the expression must match provided information. \n"
            "If any number is given, return input- 'error'. \n"
        )

    @property
    def refiner_input_format(self) -> str:
        return "<single python arithmetic expression>"

    @property
    def refiner_examples(self) -> str:
        return "\n".join(
            [
                "(2000/876785433)*100",
                "round((450/1200)*100, 2)",
                "(15/100)*800",
                "(12 + 3) * 5",
                "math.floor(3.7)",
            ]
        )

    @property
    def refiner_forbidden(self) -> str:
        return "No .format, no f-strings, no assignments, no imports, no unknown names."

    @property
    def refiner_max_chars(self) -> int:
        return self.MAX_INPUT_LENGTH

    def execute(self, input_string: str) -> Evidence:
        logger.info("[calculator] Received raw input length=%d", len(input_string or ""))

        try:
            preprocessed = self._preprocess_input(input_string)
        except Exception as e:
            logger.warning("[calculator] Preprocess failed error=%r", e)
            return self._evidence_error(str(e), error=e)

        validation_error = self._validate_input(preprocessed)
        if validation_error:
            logger.warning("[calculator] Validation failed message=%r", validation_error)
            return self._evidence_error(validation_error)

        try:
            if not self._is_safe_expression(preprocessed, allowed_names=self.ALLOWED_NAMES_DEFAULT):
                raise ValueError(self.ERROR_UNSAFE_EXPRESSION)

            tree = ast.parse(preprocessed, mode="eval")
            typing_info = self._infer_result_type(tree)

            value = self._evaluate_expression(preprocessed)

            unit = typing_info.get("unit")
            expr_kind = typing_info.get("expr_kind")
            already_scaled = bool(typing_info.get("already_scaled", False))

            ndigits = self._top_level_round_ndigits(tree)
            display_value = self._format_result(
                value=value,
                unit=unit if isinstance(unit, str) else None,
                already_scaled=already_scaled,
                ndigits=ndigits,
            )

            content = f"{preprocessed} = {display_value}"

            extracted: Dict[str, object] = {
                "expr": preprocessed,
                "result": float(value) if isinstance(value, (int, float)) else str(value),
                "display": display_value,
                "unit": unit or "",
                "expr_kind": expr_kind or "",
                "already_scaled": already_scaled,
            }
            if ndigits is not None:
                extracted["round_ndigits"] = int(ndigits)

            logger.info("[calculator] evidence.content=%r", content)

            return Evidence(
                tool=self.name,
                content=content,
                url=None,
                extracted=extracted,
                as_of=datetime.utcnow(),
                confidence=0.9,
            )

        except Exception as e:
            logger.warning("[calculator] Execution failed error=%r", e)
            return self._evidence_error(f"Error: {e}", error=e)

    def _preprocess_input(self, input_string: str) -> str:
        s = (input_string or "").strip()
        if not s:
            return s

        s = re.sub(r"\s+", " ", s)

        m = re.fullmatch(
            r"\(\s*(?P<inner>.+?)\s*\)\.format\(\s*(?P<digits>\d+)\s*\)",
            s,
            flags=re.IGNORECASE,
        )
        if m:
            inner = m.group("inner")
            digits = m.group("digits")
            return f"round(({inner}), {digits})"

        if " or " in s.lower():
            parts = re.split(r"\bor\b", s, flags=re.IGNORECASE)
            s = (parts[0] or "").strip()

        s = self._normalise_thousands_commas(s)
        return s.strip()

    def _normalise_thousands_commas(self, s: str) -> str:
        if "," not in s:
            return s

        def _strip_commas(match: re.Match) -> str:
            return match.group(0).replace(",", "")

        rewritten = self._THOUSANDS_NUMBER_RE.sub(_strip_commas, s)

        if self._ANY_DIGIT_COMMA_DIGIT_RE.search(rewritten):
            for m in self._ANY_DIGIT_COMMA_DIGIT_RE.finditer(rewritten):
                start = max(0, m.start() - 10)
                end = min(len(rewritten), m.end() + 10)
                win = rewritten[start:end]
                if not self._THOUSANDS_COMMA_POS_RE.search(win):
                    raise ValueError(self.ERROR_AMBIGUOUS_COMMAS)

        return rewritten

    def _validate_input(self, input_string: str) -> Optional[str]:
        if not input_string or not input_string.strip():
            return self.ERROR_EMPTY_INPUT
        if len(input_string.strip()) > self.MAX_INPUT_LENGTH:
            return self.ERROR_INPUT_TOO_LONG
        return None

    def _is_safe_expression(self, expression: str, allowed_names: Set[str]) -> bool:
        try:
            tree = ast.parse(expression, mode="eval")
            for node in ast.walk(tree):
                if not isinstance(node, self.ALLOWED_AST_NODES):
                    return False

                if isinstance(node, ast.Name):
                    if node.id not in allowed_names:
                        return False

                if isinstance(node, ast.Attribute):
                    if not isinstance(node.value, ast.Name) or node.value.id != "math":
                        return False
                    if node.attr not in self.SAFE_FUNCTIONS and node.attr not in self.SAFE_ATTRIBUTES:
                        return False

                if isinstance(node, ast.Call):
                    if node.keywords:
                        return False

                    if isinstance(node.func, ast.Name) and node.func.id == "round":
                        if len(node.args) != 2:
                            return False
                        arg1 = node.args[1]
                        if not isinstance(arg1, ast.Constant) or not isinstance(arg1.value, int):
                            return False

                    elif isinstance(node.func, ast.Attribute):
                        if node.func.attr not in self.SAFE_FUNCTIONS:
                            return False
                        if len(node.args) != 1:
                            return False
                    else:
                        return False

            return True
        except Exception:
            return False

    def _evaluate_expression(self, expression: str) -> Union[int, float]:
        safe_globals = {"math": math, "round": round, "__builtins__": None}
        result = eval(expression, safe_globals, {})
        if isinstance(result, float):
            result = round(result, self.FLOAT_ROUND_DECIMALS)
            if result.is_integer():
                return int(result)
        return result

    def _format_result(
        self,
        *,
        value: Union[int, float],
        unit: Optional[str],
        already_scaled: bool,
        ndigits: Optional[int],
    ) -> str:
        if unit == self.UNIT_PERCENT and already_scaled:
            return self._format_percent(value)

        if ndigits is not None and isinstance(value, (int, float)):
            return self._format_fixed_decimals(value, ndigits)

        return self._format_value(value)

    def _format_value(self, value: Union[int, float]) -> str:
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return (
                f"{value:.{self.FLOAT_FORMAT_DECIMALS}f}".rstrip("0").rstrip(".")
                or "0"
            )
        return str(value)

    def _format_fixed_decimals(self, value: Union[int, float], ndigits: int) -> str:
        nd = max(0, int(ndigits))
        try:
            d = Decimal(str(float(value)))
            q = Decimal("1").scaleb(-nd)  
            out = d.quantize(q, rounding=ROUND_HALF_UP)
            return format(out, "f")
        except (InvalidOperation, ValueError):
            return f"{float(value):.{nd}f}"

    def _format_percent(self, value: Union[int, float]) -> str:
        v = float(value)
        return f"{v:.{self.PERCENT_FORMAT_DECIMALS}f}%"

    def _top_level_round_ndigits(self, tree: ast.AST) -> Optional[int]:
        try:
            if not isinstance(tree, ast.Expression):
                return None
            body = tree.body
            if isinstance(body, ast.Call) and isinstance(body.func, ast.Name) and body.func.id == "round":
                if len(body.args) == 2 and isinstance(body.args[1], ast.Constant) and isinstance(body.args[1].value, int):
                    return int(body.args[1].value)
        except Exception:
            return None
        return None

    def _infer_result_type(self, tree: ast.AST) -> Dict[str, object]:
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
            return {"expr_kind": self.EXPR_KIND_PERCENT_RELATIVE, "unit": self.UNIT_PERCENT, "already_scaled": True}
        if has_multiplication and has_divide_by_100:
            return {"expr_kind": self.EXPR_KIND_PERCENT_OF, "unit": None, "already_scaled": False}
        if has_division and not has_multiply_by_100 and not has_divide_by_100:
            return {"expr_kind": self.EXPR_KIND_RATIO, "unit": None, "already_scaled": False}
        return {"expr_kind": self.EXPR_KIND_ARITHMETIC, "unit": None, "already_scaled": False}

    def _is_const_100(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            try:
                return float(node.value) == self.PERCENT_BASE_VALUE
            except Exception:
                return False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd) and isinstance(node.operand, ast.Constant):
            try:
                return float(node.operand.value) == self.PERCENT_BASE_VALUE
            except Exception:
                return False
        return False

    def _evidence_error(self, message: str, error: Exception = None) -> Evidence:
        extracted: Dict[str, object] = {"error": True, "message": str(message)}
        if error is not None:
            extracted["exception"] = repr(error)

        return Evidence(
            tool=self.name,
            content=str(message),
            url=None,
            extracted=extracted,
            as_of=datetime.utcnow(),
            confidence=0.1,
        )
