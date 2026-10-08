from langchain.tools import tool


@tool
def search_local_knowledge(query: str) -> str:
    """Recherche des informations dans la base de connaissances locale."""
    return (
        f"Résultat de recherche local pour : {query}\n"
        "LangGraph utilise un State partagé entre les nodes."
    )