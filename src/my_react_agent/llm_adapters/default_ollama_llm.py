from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from .llm_base import LLMBase

FormatArg = Union[None, str, Dict[str, Any]]

class DefaultOllamaLLM(LLMBase):
    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        num_ctx: int = 2048,
        stop: Tuple[str, ...] = ("<|endoftext|>",),
    ):
        self.model = str(model)
        self.temperature = float(temperature)
        self.num_ctx = int(num_ctx or 2048)
        self.stop = list(stop or ("<|endoftext|>",))

    def _normalise_format(self, fmt: FormatArg) -> Optional[Union[str, Dict[str, Any]]]:
        if fmt is None:
            return None
        if isinstance(fmt, dict):
            return fmt
        if isinstance(fmt, str) and fmt.strip().lower() == "json":
            return "json"
        return None

    def _prompt_wants_json(self, prompt: str) -> bool:
        p = (prompt or "").lower()
        markers = (
            "return strict json",
            "output strict json",
            "return only json",
            "output only json",
            "single json object",
            "json:",
            "schema",
            '"tool_input"',
            '"action"',
            '"steps"',
        )
        return any(m in p for m in markers)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            import ollama
        except Exception as e:
            raise ConnectionError(f"Ollama error: {str(e)}")

        try:
            fmt = self._normalise_format(kwargs.get("format"))

            if fmt is None and self._prompt_wants_json(prompt):
                fmt = "json"

            req: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_ctx": kwargs.get("num_ctx", self.num_ctx),
                    "stop": kwargs.get("stop", self.stop),
                },
            }

            if fmt is not None:
                req["format"] = fmt

            response = ollama.generate(**req)
            return (response.get("response") or "").strip()

        except Exception as e:
            raise ConnectionError(f"Ollama error: {str(e)}")
