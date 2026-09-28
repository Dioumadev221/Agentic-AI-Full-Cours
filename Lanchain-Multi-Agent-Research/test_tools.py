from dotenv import load_dotenv
load_dotenv()

from src.tools.tools import web_search, scrape_url

print(web_search.invoke({"query": "AI job market 2026", "max_results": 3}))
print("=" * 50)
print(scrape_url.invoke({"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"}))
print("=" * 50)
print(scrape_url.invoke({"url": "pas-une-url"}))
print("=" * 50)
print(web_search.args)   # ce que le LLM voit de ton outil