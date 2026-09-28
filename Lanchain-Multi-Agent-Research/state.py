from typing import TypedDict

from pydantic import BaseModel, Field

class Source(BaseModel):
    """A single web source found by the search agent."""

    title: str = Field(description="Title of the web page.")
    url: str = Field(description="Full URL of the web page.")


class SearchResults(BaseModel):
    """The most relevant sources found for the topic."""

    sources: list[Source] = Field(
        description="The 3 most relevant and reliable sources for the topic."
    )

class Critique(BaseModel):
    """Structured review of a research report."""

    score: int = Field(ge=0, le=10, description="Overall quality score from 0 to 10.")
    strengths: list[str] = Field(description="What the report does well.")
    improvements: list[str] = Field(description="Specific points the writer should fix.")
    verdict: str = Field(description="One-sentence overall verdict.")





class ResearchState(TypedDict):
    """Shared state passed between the nodes of the research graph."""

    topic: str               # user's question      → input of Search
    sources: list[Source]    # Search               → Scraping
    scraped_content: str     # Scraping             → Writer
    report: str              # Writer               → Critic + shown to the user
    feedback: Critique       # Critic               → shown to the developer