import ollama
import logging

class ToolReponse:
    def __init__ (self, content: str, success: bool, metadata: dict = None):
        self.content = content
        self.success = success
        self.metadata = metadata or {} 

class AgentReponse:
    def __init__ (self, response: str, tool_uses: int = 0, iteration: int = 0, sources: List [str]= None):
        self.response = esponse
        self.tool_uses = tool_uses
        self.iteration = iteration
        self.sources = sourcesor []

class Event(Enum): 
    BEFORE_TOOL_EXECUTE ="" "before_tool_execute"
    AFTER_TOOL_EXECUTE = "after_tool_execute"
    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_LLM_CALL = "after_llm_call"
    ON_ERROR = "on_error"

class TaskManager:
    def __init__(self, llm: LLMInterface, tools: Dict[str, AgentTool]):
        self.llm = llm,
        self.tools = tools
    
    def generateResponseFromLLM (self, prompt: str) -> str:
        response = ollama.generate(
            model = "deepseek-r1:8b", 
            prompt = prompt,
        )
        return response['response']

