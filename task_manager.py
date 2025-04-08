import ollama

response = ollama.generate(
    model = "deepseek-r1:8b", 
    prompt = "What is the universe?"
)

print("Response:", response['response'])
