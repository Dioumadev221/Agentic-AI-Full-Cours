from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    titre: str = Field(description="Title of the web page.")
    url: str = Field(description="URL of the web page.")


class SearchResults(BaseModel):
    sources: List[Source] = Field(description="The most relevant and reliable sources for the topic.")