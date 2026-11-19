from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from deepagents.backends import FilesystemBackend
from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware

dossier = Path(__file__).parent
fichier_memoire = dossier / "AGENTS.md"


def ecrire_memoire(ligne):
    """Recrée AGENTS.md en UTF-8 avec une seule ligne de préférence."""
    texte = "# Memoire de l'agent\n\n## Preferences\n" + ligne + "\n"
    fichier_memoire.write_bytes(texte.encode("utf-8"))


modele = ChatOllama(model="qwen2.5:3b", num_ctx=16384, temperature=0)
backend = FilesystemBackend(root_dir=str(dossier), virtual_mode=True)

agent = create_agent(
    model=modele,
    middleware=[
        FilesystemMiddleware(backend=backend),
        MemoryMiddleware(backend=backend, sources=["/AGENTS.md"]),
        ToolCallLimitMiddleware(run_limit=4),
    ],
    checkpointer=InMemorySaver(),
)


def parler(texte, thread):
    cfg = {"configurable": {"thread_id": thread}}
    res = agent.invoke({"messages": [{"role": "user", "content": texte}]}, config=cfg)
    print(f"[{thread}] {res['messages'][-1].content}\n")


def voir_etat(thread):
    cfg = {"configurable": {"thread_id": thread}}
    valeurs = agent.get_state(cfg).values
    print(f"--- memory_contents du thread {thread} ---")
    print(valeurs.get("memory_contents"))
    print()



question = "Selon ta memoire, quelle langue dois-tu utiliser ? Reponds en une phrase."

print(">> Thread A, message 1")
parler(question, "A")
voir_etat("A")

# On simule edit_file : c'est NOTRE code qui modifie le fichier
ecrire_memoire("Langue : FRANCAIS uniquement")
print(">> Le fichier vient d'etre modifie par Python\n")

print(">> Thread A, message 2 (meme conversation)")
parler(question, "A")
voir_etat("A")

print(">> Thread B, nouvelle conversation")
parler(question, "B")
voir_etat("B")

print("##### PARTIE 2 : le modele ecrit avec edit_file #####\n")

ecrire_memoire("AUCUNE_PREFERENCE")
parler(
    "Utilise l'outil edit_file sur /AGENTS.md avec old_string = AUCUNE_PREFERENCE "
    "et new_string = Nom: DIOUMA. Langue: francais. "
    "Copie ces valeurs sans les modifier.",
    "C",
)


print("=== CONTENU FINAL DE AGENTS.md ===")
print(fichier_memoire.read_text(encoding="utf-8"))

