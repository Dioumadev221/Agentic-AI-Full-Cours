import os
import re

import requests
import trafilatura
from bs4 import BeautifulSoup
from langchain.tools import tool
from readability import Document
from tavily import TavilyClient

MAX_CHARS = 5000
MIN_CONTENT_LENGTH = 200
SNIPPET_LENGTH = 300
REQUEST_TIMEOUT = 15
UNWANTED_TAGS = [
    'script',
    'style',
    'nav',
    'footer',
    'header',
    'aside',
    'form',
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

tavily_client = TavilyClient(api_key= os.environ.get("TAILLY_API_KEY"))

def _clean(text: str) -> str:
    """Collapse whitespace and truncate the text to MAX_CHARS."""
    return re.sub(r"\s+", " ", text).strip()[:MAX_CHARS]


def _html_to_text(html: str) -> str:
    """Remove unwanted tags and return the page's plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(UNWANTED_TAGS):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)



@tool(parse_docstring=True)
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for recent and reliable information on a topic.

    Args:
        query: A short, precise search query in English (3-10 words).
        max_results: Number of results to return, between 1 and 10.
    """
    try:
        response = tavily_client.search(query=query, max_results=max_results)
        results = response.get("results", [])

        if not results:
            return "No results found."

        out = []
        for r in results:
            out.append(
                f"Title: {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Snippet: {r['content'][:SNIPPET_LENGTH]}...."
            )
        return "\n----\n".join(out)

    except Exception as e:
        return f"Search failed: {e}"



@tool(parse_docstring=True)
def scrape_url(url: str) -> str:
    """Download a web page and extract its main readable content.

    Args:
        url: The full URL of the page to scrape, starting with http:// or https://.
    """
    # 0. Check the URL before doing anything
    if not url.startswith(("http://", "https://")):
        return "Invalid URL: it must start with http:// or https://."

    try:
        # 1. Download the page
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        html = response.text

        # 2. Strategy 1: trafilatura (best for articles and blogs)
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        if text and len(text) > MIN_CONTENT_LENGTH:
            return _clean(text)

        # 3. Strategy 2: readability (keeps only the main part of the page)
        main_html = Document(html).summary()
        text = _html_to_text(main_html)
        if len(text) > MIN_CONTENT_LENGTH:
            return _clean(text)

        # 4. Strategy 3: fallback on the whole page
        text = _html_to_text(html)
        if text:
            return _clean(text)

        return "Could not extract meaningful content from the page."

    except requests.exceptions.Timeout:
        return "Request timed out while scraping the URL."
    except requests.exceptions.HTTPError as e:
        return f"HTTP error: {e}"
    except Exception as e:
        return f"Could not scrape URL: {e}"