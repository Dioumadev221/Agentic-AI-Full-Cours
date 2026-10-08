from langchain_core.messages import HumanMessage, AIMessage
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



MAX_TOOL_CALLS = 5
MAX_REWRITES = 2

tools = [search_local_knowledge]
model_with_tools = model.bind_tools(tools)


def agent(state: MessagesState):
    reponse = model_with_tools.invoke(state["messages"])
    return {"messages": [reponse]}


def rewrite_question(state: MessagesState):
    """Reformule la question initiale pour améliorer la prochaine recherche."""
    question = state["messages"][0].content
    prompt = (
        "Reformule la question suivante pour améliorer une recherche "
        "sémantique. Garde le sens, change la formulation.\n\n"
        f"Question : {question}\n\n"
        "Nouvelle question :"
    )
    nouvelle = model.invoke([HumanMessage(prompt)]).content.strip()
    return {"messages": [HumanMessage(content=nouvelle)]}



def route_after_agent(state: MessagesState):
    nb_tool_msgs = sum(1 for m in state["messages"] if m.type == "tool")
    if nb_tool_msgs >= MAX_TOOL_CALLS:
        return END

    dernier = state["messages"][-1]
    if getattr(dernier, "tool_calls", None):
        return "tools"
    return END



def route_after_tools(state: MessagesState):
    """Juge silencieusement puis route : SUFFICIENT → agent | INSUFFICIENT → rewrite."""
    nb_rewrites = sum(
        1 for m in state["messages"]
        if m.type == "human" and m is not state["messages"][0]
    )
    if nb_rewrites >= MAX_REWRITES:
        return "agent"

    prompt = (
        "Tu es un évaluateur. Regarde la conversation ci-dessus et dis si "
        "les informations collectées sont SUFFISANTES pour répondre à la "
        "question initiale.\n\n"
        "Réponds UNIQUEMENT par un mot : SUFFICIENT ou INSUFFICIENT."
    )
    verdict = model.invoke(state["messages"] + [HumanMessage(prompt)]).content.strip().upper()
    print(f"🔍 verdict={verdict} | rewrites={nb_rewrites}")
    if "INSUFFICIENT" in verdict:
        return "rewrite"
    return "agent"



builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_node("rewrite", rewrite_question)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent, ["tools", END])
builder.add_conditional_edges("tools", route_after_tools, ["agent", "rewrite"])
builder.add_edge("rewrite", "agent")

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke({
        "messages": [HumanMessage("Quels sont les concepts principaux de LangGraph ?")]
    })
    print(result["messages"][-1].content)