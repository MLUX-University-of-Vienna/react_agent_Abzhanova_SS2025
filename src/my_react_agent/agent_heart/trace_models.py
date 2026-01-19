from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ReActStep:
    index: int
    thought: str
    action_kind: str
    action_tool: Optional[str]
    action_input: str
    observation: str
    evidence_snippet: Optional[str] = None
