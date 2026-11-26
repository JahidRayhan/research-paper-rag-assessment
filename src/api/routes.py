"""REST API: paper ingestion/management, querying, history and analytics."""
import logging
import os
import time
from collections import Counter

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi import Query as QueryParam
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.deps import get_cache, get_db, get_embedder, get_pipeline, get_store, require_api_key
from src.config import get_settings
from src.models.db import Excerpt, Paper, QueryLog, TopicPaper
from src.models.schemas import (PaperOut, PaperStats, QueryHistoryItem, QueryRequest, QueryResponse, RatingIn)
from src.services.analytics import extract_terms
from src.services.ingest import DuplicatePaper, IngestError, ingest_pdf

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


def _get_paper_or_404(db: Session, paper_id: int) -> Paper:
    paper = db.get(Paper, paper_id)
    if not paper:
        raise HTTPException(404, f"Paper {paper_id} not found")
    return paper


# ------------------------------------------------------------------ papers
@router.post("/papers/upload", response_model=PaperOut, status_code=201)
def upload_paper(file: UploadFile = File(...), title: str | None = Form(None), authors: str | None = Form(None),
                 year: int | None = Form(None), db: Session = Depends(get_db), embedder=Depends(get_embedder),
                 store=Depends(get_store), cache=Depends(get_cache)):
    """Upload + index a PDF. Optional form fields override auto-detected title/authors/year."""
    s = get_settings()
    name = os.path.basename(file.filename or "paper.pdf")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(415, "Only .pdf files are supported")
    data = file.file.read(s.max_upload_mb * 1024 * 1024 + 1)
    author_list = [a.strip() for a in authors.replace(";", ",").split(",") if a.strip()] if authors else None
    try:
        paper = ingest_pdf(db, embedder, store, data, name, title=title, authors=author_list, year=year)
    except DuplicatePaper as exc:
        raise HTTPException(409, detail={"message": exc.detail, "paper_id": exc.paper_id}) from exc
    except IngestError as exc:
        raise HTTPException(exc.status, exc.detail) from exc
    cache.clear()
    return paper


@router.get("/papers", response_model=list[PaperOut])
def list_papers(skip: int = QueryParam(0, ge=0), limit: int = QueryParam(50, ge=1, le=200),
                db: Session = Depends(get_db)):
    return db.query(Paper).order_by(Paper.id).offset(skip).limit(limit).all()


@router.get("/papers/{paper_id}", response_model=PaperOut)
def get_paper(paper_id: int, db: Session = Depends(get_db)):
    paper = _get_paper_or_404(db, paper_id)
    paper.view_count += 1
    db.commit()
    return paper


@router.get("/papers/{paper_id}/download")
def download_paper(paper_id: int, db: Session = Depends(get_db)):
    paper = _get_paper_or_404(db, paper_id)
    if not os.path.exists(paper.stored_path):
        raise HTTPException(410, "Original file is no longer available")
    paper.download_count += 1
    db.commit()
    return FileResponse(paper.stored_path, media_type="application/pdf", filename=paper.filename)


@router.get("/papers/{paper_id}/stats", response_model=PaperStats)
def paper_stats(paper_id: int, db: Session = Depends(get_db)):
    p = _get_paper_or_404(db, paper_id)
    return PaperStats(paper_id=p.id, title=p.title, view_count=p.view_count, download_count=p.download_count,
                      times_cited=p.query_count, num_pages=p.num_pages, num_chunks=p.num_chunks)


@router.delete("/papers/{paper_id}")
def delete_paper(paper_id: int, db: Session = Depends(get_db), store=Depends(get_store), cache=Depends(get_cache)):
    paper = _get_paper_or_404(db, paper_id)
    store.delete_paper(paper_id)  # vectors first: a DB row without vectors is recoverable, the reverse is not
    path = paper.stored_path
    db.query(TopicPaper).filter(TopicPaper.paper_id == paper_id).delete()
    db.query(Excerpt).filter(Excerpt.paper_id == paper_id).delete()
    db.delete(paper)
    db.commit()
    if path and os.path.exists(path):
        os.remove(path)
    cache.clear()
    return {"deleted": paper_id, "vectors_removed": True}


# ------------------------------------------------------------------ query
@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, db: Session = Depends(get_db), pipeline=Depends(get_pipeline),
          cache=Depends(get_cache)):
    t0 = time.perf_counter()
    known = db.query(Paper.id, Paper.title).all()
    if not known:
        raise HTTPException(409, "No papers indexed yet - upload a PDF first")
    if req.paper_ids:
        missing = sorted(set(req.paper_ids) - {k[0] for k in known})
        if missing:
            raise HTTPException(404, f"Paper(s) not found: {missing}")

    key = (" ".join(req.question.lower().split()), req.top_k, tuple(sorted(req.paper_ids or [])))
    result = cache.get(key)
    cached = result is not None
    if not cached:
        result = pipeline.answer(req.question, req.top_k, req.paper_ids, [(k[0], k[1]) for k in known])
        cache.set(key, result)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    cited_ids = sorted({c["paper_id"] for c in result["citations"]})
    row = QueryLog(question=req.question, answer=result["answer"], paper_ids=cited_ids,
                   sources=result["sources_used"], top_k=req.top_k, confidence=result["confidence"],
                   response_time_ms=elapsed, llm_provider=result["llm_provider"], cached=cached)
    db.add(row)
    for pid in cited_ids:
        db.query(Paper).filter(Paper.id == pid).update({Paper.query_count: Paper.query_count + 1})
    db.commit()
    return QueryResponse(query_id=row.id, response_time_ms=elapsed, cached=cached, **result)


@router.get("/queries/history", response_model=list[QueryHistoryItem])
def query_history(limit: int = QueryParam(20, ge=1, le=200), offset: int = QueryParam(0, ge=0),
                  db: Session = Depends(get_db)):
    return db.query(QueryLog).order_by(QueryLog.id.desc()).offset(offset).limit(limit).all()


@router.post("/queries/{query_id}/rating")
def rate_query(query_id: int, body: RatingIn, db: Session = Depends(get_db)):
    row = db.get(QueryLog, query_id)
    if not row:
        raise HTTPException(404, f"Query {query_id} not found")
    row.rating = body.rating
    db.commit()
    return {"query_id": query_id, "rating": body.rating}


@router.get("/analytics/popular")
def popular(limit: int = QueryParam(10, ge=1, le=50), db: Session = Depends(get_db)):
    recent = db.query(QueryLog.question).order_by(QueryLog.id.desc()).limit(1000).all()
    counter: Counter = Counter()
    for (q,) in recent:
        counter.update(set(extract_terms(q)))  # count once per query
    topics = [{"term": t, "count": c} for t, c in counter.most_common(limit * 3) if c >= 1]
    # prefer bigrams when they are as frequent as their unigrams (more descriptive)
    topics = sorted(topics, key=lambda x: (-x["count"], -len(x["term"].split())))[:limit]
    top_papers = db.query(Paper).order_by(Paper.query_count.desc()).limit(limit).all()
    return {
        "total_queries": db.query(func.count(QueryLog.id)).scalar() or 0,
        "avg_response_time_ms": round(db.query(func.avg(QueryLog.response_time_ms)).scalar() or 0, 1),
        "avg_rating": (lambda v: round(v, 2) if v else None)(db.query(func.avg(QueryLog.rating)).scalar()),
        "popular_topics": topics,
        "most_cited_papers": [{"paper_id": p.id, "title": p.title, "times_cited": p.query_count} for p in top_papers],
    }
