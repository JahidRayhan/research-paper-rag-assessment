"""SQLAlchemy models + session handling (SQLite by default, Postgres/MySQL via DATABASE_URL)."""
import os
from datetime import datetime, timezone

from sqlalchemy import (JSON, DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine,
                        inspect, text)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    authors: Mapped[list] = mapped_column(JSON, default=list)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    stored_path: Mapped[str] = mapped_column(String(500))
    num_pages: Mapped[int] = mapped_column(Integer, default=0)
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    sections: Mapped[list] = mapped_column(JSON, default=list)
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    download_count: Mapped[int] = mapped_column(Integer, default=0)
    query_count: Mapped[int] = mapped_column(Integer, default=0)  # times cited in answers
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # bibliographic metadata (filled for papers found via discovery; all optional)
    source: Mapped[str | None] = mapped_column(String(20), default="upload", nullable=True)
    doi: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(60), nullable=True, index=True)
    openalex_id: Mapped[str | None] = mapped_column(String(60), nullable=True)
    url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    venue: Mapped[str | None] = mapped_column(String(300), nullable=True)
    citation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Topic(Base):
    """A research topic: a named bundle of papers, saved excerpts and summaries."""
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    query: Mapped[str | None] = mapped_column(Text, nullable=True)
    candidates: Mapped[list] = mapped_column(JSON, default=list)  # discovered-but-not-yet-imported papers
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TopicPaper(Base):
    __tablename__ = "topic_papers"
    __table_args__ = (UniqueConstraint("topic_id", "paper_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(Integer, index=True)
    paper_id: Mapped[int] = mapped_column(Integer, index=True)
    relevance: Mapped[float] = mapped_column(Float, default=0.0)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Excerpt(Base):
    """An exact passage/section pulled from a paper for a topic (your reading notes)."""
    __tablename__ = "excerpts"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(Integer, index=True)
    paper_id: Mapped[int] = mapped_column(Integer, index=True)
    section: Mapped[str] = mapped_column(String(100))
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    query: Mapped[str] = mapped_column(Text, default="")
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(Integer, index=True)
    focus: Mapped[str] = mapped_column(Text, default="")
    markdown: Mapped[str] = mapped_column(Text)
    paper_ids: Mapped[list] = mapped_column(JSON, default=list)
    llm_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    paper_ids: Mapped[list] = mapped_column(JSON, default=list)  # papers actually cited
    sources: Mapped[list] = mapped_column(JSON, default=list)
    top_k: Mapped[int] = mapped_column(Integer, default=5)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    response_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    llm_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    cached: Mapped[bool] = mapped_column(default=False)
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


_engine = None
_SessionLocal = None


def init_db(url: str) -> None:
    global _engine, _SessionLocal
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        path = url.split("///", 1)[-1]
        if path and path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _engine = create_engine(url, pool_pre_ping=True, **kwargs)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    Base.metadata.create_all(_engine)
    _add_missing_columns(_engine)


def _add_missing_columns(engine) -> None:
    """Tiny migration: add columns introduced after a database was first created (all nullable)."""
    insp = inspect(engine)
    for table in Base.metadata.sorted_tables:
        existing = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name not in existing:
                ddl = col.type.compile(dialect=engine.dialect)
                with engine.begin() as conn:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl}'))


def get_engine():
    return _engine


def get_db():
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
