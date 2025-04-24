import ollama


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
    
    def build_initial_prompt (self, query: str) -> str:
        tool_descriptions = "\n".join(
            f"{tool.name}: {tool.description}" for tool in self.tools.values()
        )

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

    def parse_response(selff, reponse: str) -> Dict[str, str]:
        thought = ""
        action = ""
        action_input = ""

        thought_match = re.search(r'Thought:\s*(.*?)(?:\nAction:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        action_match = re.search(r'Action:\s*(\w+)', response)
        if action_match:
            action = action_match.group(1).strip()
        
        input_match = re.search(r'Action Input:\s*(.*?)(?:\n|$)', response, re.DOTALL)
        if input_match:
            action_input = input_match.group(1).strip()

        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "final_answer": "Final Answer:" in response
        }

    def run(self, query: str, max_iters: int = 5) -> str:
        prompt = self._build_react_prompt(query)
        history = []
        
        for _ in range(max_iters):
            response = self.generate(prompt)
            parsed = self.parse_response(response)
            
            if parsed["final_answer"]:
                return response.split("Final Answer:")[-1].strip()
                
            if parsed["action"] and parsed["action"] in self.tools:
                tool_output = self.tools[parsed["action"]](parsed["action_input"])
                prompt += f"\nObservation: {tool_output}"
            else:
                prompt += "\nObservation: Invalid action or missing action input"
                
        return "Maximum iterations reached without final answer."
