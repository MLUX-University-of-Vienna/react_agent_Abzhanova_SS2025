import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from llm_interface.ollama_interface import OllamaInterface
from tool_management.calculator_tool import CalculatorTool
from task_management.task_manager import TaskManager

llm = OllamaInterface(model="deepseek-r1:8b")
calculator = CalculatorTool()
tools = {"calculator": CalculatorTool()}

agent = TaskManager(llm=llm, tools=tools)

query = "What is (15 * 3 + 2) / 5?"
result = agent.run(query)

print("\n--- FINAL ANSWER ---")
print(result["final_answer"])

print("\n--- FULL OUTPUT ---")
print(result["full_output"])
