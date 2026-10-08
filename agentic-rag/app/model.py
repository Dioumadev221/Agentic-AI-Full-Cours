from langchain_ollama import ChatOllama


model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)