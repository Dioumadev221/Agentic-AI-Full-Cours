from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

class Answer(BaseModel):
    """Réponse finale de l'agent."""
    summary: str = Field(description="Résumé en 3 phrases max")
    confidence: float = Field(description="Confiance entre 0 et 1", ge=0, le=1)

# --- Agent A : ToolStrategy ---
agent_a = create_agent(
    model="ollama:qwen2.5:3b",
    response_format=ToolStrategy(Answer),
)

result_a = agent_a.invoke({
    "messages": [{"role": "user", "content": "Explique en bref ce qu'est un agent IA."}]
})

print("=== Agent A (ToolStrategy) ===")
print("Contenu :", result_a["structured_response"])



# --- Agent B : ProviderStrategy ---
agent_b = create_agent(
    model="ollama:qwen2.5:3b",
    response_format=ProviderStrategy(Answer),
)

result_b = agent_b.invoke({
    "messages": [{"role": "user", "content": "Explique en bref ce qu'est un agent IA."}]
})

print("\n=== Agent B (ProviderStrategy) ===")
print("Contenu :", result_b["structured_response"])