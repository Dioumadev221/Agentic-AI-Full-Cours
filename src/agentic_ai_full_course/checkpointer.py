from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class Context:
    user_id: str


agent = create_agent(
    model="ollama:qwen2.5:3b",
    tools=[],
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

# Générer le thread_id UNE SEULE FOIS
thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is DIOUMA"}]},
    config=config,
    context=Context(user_id="user-123"),
)
print(result["messages"][-1].content)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    config=config,
    context=Context(user_id="user-123"),
)
print("**************************************************************")
print(result["messages"][-1].content)