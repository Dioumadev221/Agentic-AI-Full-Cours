from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode

from .model import model
from .tools import search_local_knowledge
# def agent(state: MessagesState):
#     """Appelle le LLM avec les messages actuels."""
#     reponse = model.invoke(state["messages"])
#     return {"messages": [reponse]}
#
#
# build = StateGraph(MessagesState)
# build.add_node("agent", agent)
# build.add_edge(START,"agent")
# build.add_edge("agent", END)
#
# graph= build.compile()
#
#
# if __name__ == "__main__":
#     result = graph.invoke({
#         "messages":[HumanMessage("presente toi .")]
#     })
#
#     print(result["messages"][-1].content)


#************************************************************************************************




# 1. On déclare les tools et on les lie au modèle
tools = [search_local_knowledge]
model_with_tools = model.bind_tools(tools)


# 2. Nœud agent : le LLM peut répondre OU demander un tool
def agent(state: MessagesState):
    reponse = model_with_tools.invoke(state["messages"])
    return {"messages": [reponse]}


# 3. Routeur : après l'agent, où va-t-on ?
MAX_TOOL_CALLS = 5

def route_after_agent(state: MessagesState):
    dernier = state["messages"][-1]

    #Garde-fou : on compte combien de ToolMessage sont déjà dans le state
    nb_tool_msgs = sum(
        1 for m in state["messages"] if m.type == "tool"
    )
    if nb_tool_msgs >= MAX_TOOL_CALLS:
        print(f"Limite {MAX_TOOL_CALLS} atteinte, on force l'arrêt.")
        return END

    # Logique normale
    if getattr(dernier, "tool_calls", None):
        return "tools"
    return END


# 4. Construction du graphe
builder = StateGraph(MessagesState)

builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent, ["tools", END])
builder.add_edge("tools", "agent")          # ⬅ la boucle

graph = builder.compile()


# 5. Test
if __name__ == "__main__":
    result = graph.invoke({
        "messages": [HumanMessage("Quels sont les concepts principaux de LangGraph ?")]
    })

    print(result["messages"][-1].content)