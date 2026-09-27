import os
import certifi
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

WEATHERSTACK_API_KEY = os.environ.get("WEATHERSTACK_API_KEY")


@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather information for a city."""
    response = requests.get(
        "https://api.weatherstack.com/current",
        params={"access_key": WEATHERSTACK_API_KEY, "query": city},
        timeout=10,
    )
    data = response.json()

    if "current" not in data:
        error = data.get("error", {}).get("info", "unknown error")
        return f"Could not fetch weather data for {city}: {error}"

    current = data["current"]
    return (
        f"City: {city}\n"
        f"Temperature: {current['temperature']}°C\n"
        f"Weather: {current['weather_descriptions'][0]}\n"
        f"Humidity: {current['humidity']}%"
    )


# Streamlit relance tout le script à chaque interaction :
# @st.cache_resource crée l'agent une seule fois et le réutilise
@st.cache_resource
def load_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    tools = [TavilySearch(max_results=2), get_weather_data]
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Use the tools when you need up-to-date information.",
    )


agent = load_agent()

st.title("🤖 Agent IA : recherche web & météo")

# st.session_state garde l'historique entre deux relances du script
if "messages" not in st.session_state:
    st.session_state.messages = []

# Réafficher la conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Pose ta question…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("L'agent réfléchit…"):
            # On envoie tout l'historique pour que l'agent se souvienne de la conversation
            response = agent.invoke({"messages": st.session_state.messages})
        answer = response["messages"][-1].text
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
