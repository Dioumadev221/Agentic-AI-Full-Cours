from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, wrap_model_call
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from deepagents.backends import FilesystemBackend
from deepagents.middleware import FilesystemMiddleware, SkillsMiddleware

dossier = Path(__file__).parent

# --- On crée la fiche (UTF-8) : /skills/rapport-reunion/SKILL.md ---
dossier_skill = dossier / "skills" / "rapport-reunion"
dossier_skill.mkdir(parents=True, exist_ok=True)
(dossier_skill / "SKILL.md").write_bytes(
    """---
name: rapport-reunion
description: Ecrire un rapport de reunion. A utiliser quand l'utilisateur demande un rapport ou un compte rendu de reunion.
---

# Rapport de reunion

Suis ces regles :
1. La premiere ligne de ta reponse est exactement : RAPPORT-OK-42
2. Ensuite, ecris 3 puces : Decisions, Actions, Prochaine reunion.
""".encode("utf-8")
)


@wrap_model_call
def espion(request, handler):
    sp = getattr(request, "system_message", None) or getattr(request, "system_prompt", None)
    texte = str(getattr(sp, "content", sp))
    i = texte.find("Skills")
    if i >= 0:
        print("   [espion] bloc Skills dans le prompt :")
        print(texte[i:i + 700].replace("\\n", "\n"))
    else:
        print("   [espion] aucun bloc Skills trouve dans le prompt")
    return handler(request)


modele = ChatOllama(model="qwen2.5:3b", num_ctx=16384, temperature=0)
backend = FilesystemBackend(root_dir=str(dossier), virtual_mode=True)

agent = create_agent(
    model=modele,
    middleware=[
        FilesystemMiddleware(backend=backend),                    # donne read_file
        SkillsMiddleware(backend=backend, sources=["/skills/"]),  # sommaire des skills
        ToolCallLimitMiddleware(run_limit=4),
        espion,                                                   # en dernier
    ],
    checkpointer=InMemorySaver(),
)


def parler(texte, thread):
    cfg = {"configurable": {"thread_id": thread}}
    res = agent.invoke({"messages": [{"role": "user", "content": texte}]}, config=cfg)
    print(f"[{thread}] {res['messages'][-1].content}\n")


def trace(thread):
    print(f"=== TRACE du thread {thread} ===")
    cfg = {"configurable": {"thread_id": thread}}
    for m in agent.get_state(cfg).values["messages"]:
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                print(f"  [appel] {tc['name']} {tc['args']}")
        elif m.type == "tool":
            print(f"  [resultat] {str(m.content)[:120]}")
        else:
            print(f"  [{m.type}] {str(m.content)[:200]}")
    print()


print("##### Test 1 : demande qui correspond a la skill #####")
parler("Fais-moi un rapport de reunion sur le lancement du produit.", "S1")
trace("S1")

print("##### Test 2 : demande sans rapport avec la skill #####")
parler("Quelle est la capitale du Senegal ?", "S2")
trace("S2")