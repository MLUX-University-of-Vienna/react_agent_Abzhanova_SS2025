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
        self.sources = sources []

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

    def initialise_tools(self) -> Dict [str, Callable]:
        return{
            "calculator": self.calculator
        }
    
    def build_initial_prompt (self, query: str) -> str:
        tool_descriptions = "calculator"

        return f"""You are a ReAct agent designed to solve complex problems. Follow these steps:

                1. Think step-by-step about how to solve the problem
                2. Use tools when needed (one per action)
                3. Combine information to form a final answer

            Available Tools:
                {tool_descriptions}

            Format for tool use:
                Thought: Your reasoning
                Action: tool_name
                Action Input: input_for_tool

            When you have the final answer:
            Final Answer: Your complete response

            Begin!

            Question: {query}"""