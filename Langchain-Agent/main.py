import os
import certifi
import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

WEATHERSTACK_API_KEY = os.environ.get("WEATHERSTACK_API_KEY")

search_tool = TavilySearch(max_results=2)

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


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools = [search_tool, get_weather_data]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use the tools when you need up-to-date information.",
)

question = "Find the capital of India and then find its current weather."

response = agent.invoke({"messages": [{"role": "user", "content": question}]})

print(response["messages"][-1].text)
