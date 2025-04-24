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

    def parse_response(self, response: str) -> Dict[str, str]:
        thought = re.search(r'Thought:\s*(.*?)(?:\nAction:|$)', response, re.DOTALL)
        action = re.search(r'Action:\s*(\w+)', response)
        action_input = re.search(r'Action Input:\s*(.*?)(?:\n|$)', response, re.DOTALL)
        final_answer = re.search(r'Final Answer:\s*(.*)', response, re.DOTALL)

        return {
            "thought": thought.group(1).strip() if thought else "",
            "action": action.group(1).strip() if action else "",
            "action_input": action_input.group(1).strip() if action_input else "",
            "final_answer": final_answer.group(1).strip() if final_answer else None
        }

