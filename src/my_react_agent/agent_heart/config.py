from dataclasses import dataclass


@dataclass
class ReActAgentConfig:
    max_steps: int = 20
    max_recent_steps_in_prompt: int = 30
    max_evidence_snippets: int = 20
    observation_snippet_limit: int = 800
    fact_schema_max_items: int = 20

    tool_confidence_threshold: float = 0.7

    step_confidence_threshold: float = 0.7
    max_step_retries: int = 10
    assess_all_steps: bool = True
