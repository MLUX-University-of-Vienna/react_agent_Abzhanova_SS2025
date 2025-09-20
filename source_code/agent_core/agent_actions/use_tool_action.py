from typing import List, Optional, Dict
import logging
from agent_core.agent_actions.action import Action
from tool_management.tools.agent_tool import AgentTool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class UseToolAction(Action):
    def __init__(self, available_tools: Optional[List[str]] = None, tool_registry: Optional[Dict[str, AgentTool]] = None):
        self._available_tools = list(available_tools or [])
        self._tool_registry = dict(tool_registry or {})
        logger.info(f"[UseToolAction.__init__] Initialised UseToolAction with {len(self._available_tools)} available_tools and registry_size={len(self._tool_registry)}")
        if self._available_tools:
            logger.debug(f"[UseToolAction.__init__] Tool names preview={self._available_tools[:8]}")
        logger.debug("[UseToolAction.__init__] This action is calibrated for single-subject, single-fact, or already-parameterised calculations only")

    @property
    def name(self) -> str:
        value = "USE_TOOL"
        logger.debug(f"[UseToolAction.name] Returning action name={value}")
        return value

    @property
    def instructions(self) -> str:
        text = (
            "Invoke one tool to compute, retrieve, or verify exactly one atomic fact or value about a single subject. "
            "Provide a concrete 'tool_name' and a minimal, deterministic 'tool_input' that follows the chosen tool's input format and regex if defined. "
            "Do not bundle multiple facts, entities, subjects, or websites in one call; avoid conjunctions, lists, logical operators, pipes, or unions. "
            "If more than one fact is required, or any operand for a calculation is missing, first choose the action that splits the work into subqueries. "
            "Use a calculator only when all required numerals are already present in the user request or have appeared in prior observations; do not introduce numbers. "
            "Treat the tool's output as the source of truth and keep 'tool_input' free of prose, extraneous filters, and unrelated constraints."
        )
        logger.debug(f"[UseToolAction.instructions] Instructions length={len(text)}")
        return text

    @property
    def when_to_pick(self) -> str:
        text = (
            "Choose this when a single deterministic tool call can answer the request for one subject, or when a single calculation is required and all operands are already known as numerals. "
            "Prefer this for atomic lookups, single definitions, single current metrics, or a straightforward calculation that does not depend on fetching additional facts."
        )
        logger.debug(f"[UseToolAction.when_to_pick] when_to_pick length={len(text)}")
        return text

    @property
    def when_to_pick_head(self) -> Optional[List[str]]:
        cues = [
            "single attribute of one entity",
            "single current value for one metric",
            "one calculation with all operands already observed",
            "simple definition or title-based lookup",
            "direct numeric transform without fetching new data",
            "single-date or single-point retrieval",
            "operator present but numerals already provided"
        ]
        logger.debug(f"[UseToolAction.when_to_pick_head] cues_count={len(cues)} cues={cues}")
        return cues

    @property
    def examples(self) -> List[str]:
        items = [
            "Use_Tool[calculator :: <single valid expression using known numerals>]",
            "Use_Tool[wikipedia :: <Exact Page Title>]",
            "Use_Tool[web_search :: web: <single focused query>]",
            "Use_Tool[vector_search :: <short keywords for one topic>]"
        ]
        logger.debug(f"[UseToolAction.examples] examples_count={len(items)} items={items}")
        return items
