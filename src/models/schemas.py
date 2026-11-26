"""Pydantic request/response schemas."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PaperOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    authors: list[str]
    year: int | None
    filename: str
    num_pages: int
    num_chunks: int
    sections: list[str]
    created_at: datetime


class PaperStats(BaseModel):
    paper_id: int
    title: str
    view_count: int
    download_count: int
    times_cited: int
    num_pages: int
    num_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    paper_ids: list[int] | None = None


class Citation(BaseModel):
    paper_id: int
    paper_title: str
    section: str
    page: int
    page_end: int
    relevance_score: float
    excerpt: str


class QueryResponse(BaseModel):
    query_id: int | None = None
    answer: str
    citations: list[Citation]
    sources_used: list[str]
    confidence: float
    response_time_ms: float
    llm_provider: str | None = None
    cached: bool = False


class RatingIn(BaseModel):
    rating: int = Field(ge=1, le=5)


class QueryHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    question: str
    answer: str
    paper_ids: list[int]
    sources: list[str]
    confidence: float
    response_time_ms: float
    llm_provider: str | None
    cached: bool
    rating: int | None
    created_at: datetime
