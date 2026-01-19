from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

@dataclass(frozen=True)
class TranscriptFormatter:
    format_action_line: Optional[Callable[..., str]]
    format_observation_line: Optional[Callable[..., str]]

    def append_action(
        self,
        transcript_lines: List[str],
        step_index: int,
        action_name: str,
        tool_name: str,
        tool_input: str,
        *,
        message: Optional[str] = None,
        final_answer: Optional[str] = None,
    ) -> None:
        if self.format_action_line:
            transcript_lines.append(
                self.format_action_line(
                    step_index,
                    action_name,
                    tool_name,
                    tool_input,
                    message=message,
                    final_answer=final_answer,
                )
            )
        else:
            transcript_lines.append(f"Act {step_index}: {action_name}")

    def append_observation(self, transcript_lines: List[str], step_index: int, observation_text: str) -> None:
        if self.format_observation_line:
            transcript_lines.append(self.format_observation_line(step_index, observation_text))
        else:
            transcript_lines.append(f"Observation {step_index}: {observation_text}")
