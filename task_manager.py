import ollama

class ToolReponse:
    def __init__ (self, content: str, success: bool, metadata: dict = None):
        self.content = content
        self.success = success
        self.metadata = metadata or {} 

class TaskManager:
    def __init__(self, llm: LLMInterface, tools: Dict[str, AgentTool]):
        self.llm = llm,
        self.tools = tools



"""
response = ollama.generate(
    model = "deepseek-r1:8b", 
    prompt = "What is the universe?"
)

print("Response:", response['response'])
"""
