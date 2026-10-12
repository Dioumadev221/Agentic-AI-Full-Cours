import os
from pathlib import Path

from bs4 import BeautifulSoup
from langchain.tools import tool
import requests
from tavily import TavilyClient
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

api_key = os.environ.get("TAVILY_API_KEY")
if not api_key:
    raise ValueError(
        f"TAVILY_API_KEY introuvable. Vérifie que le fichier existe : {env_path}"
    )

tavily_client = TavilyClient(api_key=api_key)
@tool
def search_local_knowledge(query: str) -> str:
    """Recherche des informations dans la base de connaissances locale."""
    return (
        f"Résultat de recherche local pour : {query}\n"
        "LangGraph utilise un State partagé entre les nodes."
    )


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Recherche des informations récentes sur le web.
    Utilise ce tool quand tu as besoin de données à jour
    ou quand la base locale ne suffit pas."""

    try:
        reponse = tavily_client.search(query, max_results=max_results)
        resultats = reponse["results"]
        if not resultats:
            return "Result Not found"

        out = []
        for result in resultats:
            out.append(
                f"Title :{result['title']}\n"
                f"URL : {result['url']}\n")
        return "\n----\n".join(out)

    except Exception as e:
        return f"Search failed: {e}"



def scrape_url(url: str) -> str:
    """Récupère le contenu textuel d'une page web à partir de son URL.
       Utilise ce tool APRÈS web_search pour lire le contenu d'une page."""
    if not url.startswith(("http://", "https://")):
        return "Invalid URL: it must start with http:// or https://."
    try:
        reponse = requests.get(url, timeout=20)
        reponse.raise_for_status()
        soup = BeautifulSoup(reponse.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        texte = soup.get_text(separator="\n", strip=True)
        return texte[:3000]

    except Exception as e:
        return f"Search failed: {e}"

