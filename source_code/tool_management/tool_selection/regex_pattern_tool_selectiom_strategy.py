import re
from llm.llm_interface import LLMInterface

class RegexPatternToolSelector(ToolSelectionStrategy):
    def __init__(self, llm: LLMInterface):
        self.llm = llm