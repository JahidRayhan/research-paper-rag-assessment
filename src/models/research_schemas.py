"""Schemas for topic research endpoints."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Source = Literal["arxiv", "openalex"]


class DiscoverRequest(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    limit: int = Field(default=15, ge=1, le=50)
    sources: list[Source] = ["arxiv", "openalex"]


class CandidateOut(BaseModel):
    index: int | None = None
    title: str
    authors: list[str] = []
    year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    venue: str | None = None
    citation_count: int | None = None
    source: str = ""
    relevance: float = 0.0
    found_via: str | None = None
    status: str | None = None        # new | indexed | imported | no_pdf | failed
    paper_id: int | None = None


class DiscoverResponse(BaseModel):
    candidates: list[CandidateOut]
    warnings: list[str] = []


class TopicCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    query: str | None = Field(default=None, max_length=300)


class TopicOut(BaseModel):
    id: int
    name: str
    query: str | None
    num_papers: int
    num_candidates: int
    num_excerpts: int
    created_at: datetime


class TopicSearchRequest(BaseModel):
    query: str | None = Field(default=None, min_length=3, max_length=300)
    limit: int = Field(default=15, ge=1, le=50)
    sources: list[Source] = ["arxiv", "openalex"]


class ImportRequest(BaseModel):
    indexes: list[int] | None = None
    top_n: int | None = Field(default=None, ge=1, le=15)


class ImportResult(BaseModel):
    index: int
    title: str
    status: str
    paper_id: int | None = None
    detail: str | None = None


class SectionsRequest(BaseModel):
    query: str | None = Field(default=None, min_length=3, max_length=300)
    per_paper: int = Field(default=2, ge=1, le=5)
    full_section: bool = True
    paper_ids: list[int] | None = None
    min_score: float = Field(default=0.2, ge=0, le=1)
    save: bool = True


class PassageOut(BaseModel):
    paper_id: int
    paper_title: str
    section: str
    page_start: int
    page_end: int
    score: float
    text: str
    matched_chunks: int
    truncated: bool
    excerpt_id: int | None = None


class ExcerptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    topic_id: int
    paper_id: int
    section: str
    page_start: int
    page_end: int
    text: str
    score: float
    query: str
    note: str | None
    created_at: datetime


class NoteIn(BaseModel):
    note: str = Field(max_length=2000)


class ExpandRequest(BaseModel):
    direction: Literal["references", "citations"]
    limit: int = Field(default=20, ge=1, le=50)
    paper_ids: list[int] | None = None


class SummaryRequest(BaseModel):
    focus: str | None = Field(default=None, min_length=3, max_length=300)
    max_papers: int = Field(default=5, ge=1, le=10)
    per_paper: int = Field(default=2, ge=1, le=4)
    style: Literal["overview", "compare", "brief"] = "overview"


class SummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    topic_id: int
    focus: str
    markdown: str
    paper_ids: list[int]
    llm_provider: str | None
    created_at: datetime
