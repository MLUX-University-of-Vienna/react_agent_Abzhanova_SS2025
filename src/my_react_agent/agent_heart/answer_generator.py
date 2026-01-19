import re

from llm_adapters.llm_base import LLMBase

class AnswerGenerator:
    def __init__(self, llm: LLMBase):
        self.llm = llm

    def generate(self, query: str, result: str) -> str:
        prompt = f"""Based on this result, answer the query:

Query: {query}
Result: {result}

Format:
 Answer: <your answer here>"""
        return self.llm.generate(prompt).strip()

    def fallback(self, query: str) -> str:
        prompt = f"""Unable to get a confident answer via tools.
Please answer this directly:

Question: {query}

Answer:"""
        return self.llm.generate(prompt).strip()