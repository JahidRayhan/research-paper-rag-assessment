"""Topic research API: discover papers, import them, pull matching sections, track references, summarise."""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Query as QueryParam
from fastapi.responses import PlainTextResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.deps import get_cache, get_db, get_embedder, get_pipeline, get_store, require_api_key
from src.config import get_settings
from src.models.db import Excerpt, Paper, Summary, Topic, TopicPaper
from src.models.research_schemas import (CandidateOut, DiscoverRequest, DiscoverResponse, ExcerptOut, ExpandRequest,
                                         ImportRequest, ImportResult, NoteIn, PassageOut, SectionsRequest,
                                         SummaryOut, SummaryRequest, TopicCreate, TopicOut, TopicSearchRequest)
from src.services import biblio, discovery, research
from src.services.ingest import DuplicatePaper, IngestError, ingest_pdf, safe_filename

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)], tags=["research"])


# ------------------------------------------------------------------ helpers
def _topic(db: Session, topic_id: int) -> Topic:
    t = db.get(Topic, topic_id)
    if not t:
        raise HTTPException(404, f"Topic {topic_id} not found")
    return t


def _topic_out(db: Session, t: Topic) -> TopicOut:
    return TopicOut(
        id=t.id, name=t.name, query=t.query, created_at=t.created_at, num_candidates=len(t.candidates or []),
        num_papers=db.query(func.count(TopicPaper.id)).filter(TopicPaper.topic_id == t.id).scalar() or 0,
        num_excerpts=db.query(func.count(Excerpt.id)).filter(Excerpt.topic_id == t.id).scalar() or 0)


def _topic_papers(db: Session, topic_id: int) -> list[Paper]:
    return (db.query(Paper).join(TopicPaper, TopicPaper.paper_id == Paper.id)
            .filter(TopicPaper.topic_id == topic_id).order_by(TopicPaper.relevance.desc(), Paper.id).all())


def _find_existing(db: Session, c: discovery.Candidate) -> Paper | None:
    for p in db.query(Paper).all():
        if (c.arxiv_id and p.arxiv_id == c.arxiv_id) or (c.doi and p.doi and p.doi.lower() == c.doi.lower()) \
                or (c.openalex_id and p.openalex_id == c.openalex_id) \
                or discovery.norm_title(p.title) == discovery.norm_title(c.title):
            return p
    return None


def _link(db: Session, topic_id: int, paper_id: int, relevance: float) -> None:
    if not db.query(TopicPaper).filter_by(topic_id=topic_id, paper_id=paper_id).first():
        db.add(TopicPaper(topic_id=topic_id, paper_id=paper_id, relevance=relevance))


def _cand_out(i: int, d: dict) -> CandidateOut:
    return CandidateOut(index=i, **{k: v for k, v in d.items() if k in CandidateOut.model_fields and k != "index"})


def _paper_dict(p: Paper, sections_used: list[str] | None = None) -> dict:
    return {"id": p.id, "title": p.title, "authors": p.authors or [], "year": p.year, "venue": p.venue, "doi": p.doi,
            "arxiv_id": p.arxiv_id, "url": p.url, "sections_used": sections_used or []}


def _sections_used(db: Session, topic_id: int) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for e in db.query(Excerpt).filter(Excerpt.topic_id == topic_id).order_by(Excerpt.id):
        pages = e.page_start if e.page_start == e.page_end else f"{e.page_start}-{e.page_end}"
        label = f"{e.section} p.{pages}"
        if label not in out.setdefault(e.paper_id, []):
            out[e.paper_id].append(label)
    return out


def _save_excerpts(db: Session, topic_id: int, query: str, passages: list[dict], overwrite: bool = True) -> None:
    for ps in passages:
        row = db.query(Excerpt).filter_by(topic_id=topic_id, paper_id=ps["paper_id"], section=ps["section"]).first()
        if row is not None and not overwrite:
            continue  # keep the exact-section text you already saved
        if row is None:
            row = Excerpt(topic_id=topic_id, paper_id=ps["paper_id"], section=ps["section"])
            db.add(row)
        row.page_start, row.page_end, row.text, row.score, row.query = (
            ps["page_start"], ps["page_end"], ps["text"], ps["score"], query)
        db.flush()
        ps["excerpt_id"] = row.id


# ------------------------------------------------------------------ discovery (stateless)
@router.post("/discover", response_model=DiscoverResponse)
def discover(req: DiscoverRequest, db: Session = Depends(get_db), embedder=Depends(get_embedder)):
    """Search arXiv/OpenAlex for a topic without saving anything."""
    cands, warnings = discovery.search_all(req.query, req.limit, req.sources, get_settings(), embedder)
    out = []
    for i, c in enumerate(cands):
        ex = _find_existing(db, c)
        d = c.to_dict() | {"status": "indexed" if ex else "new", "paper_id": ex.id if ex else None}
        out.append(_cand_out(i, d))
    return DiscoverResponse(candidates=out, warnings=warnings)


# ------------------------------------------------------------------ topics
@router.post("/topics", response_model=TopicOut, status_code=201)
def create_topic(body: TopicCreate, db: Session = Depends(get_db)):
    t = Topic(name=body.name.strip(), query=(body.query or body.name).strip(), candidates=[])
    db.add(t)
    db.commit()
    return _topic_out(db, t)


@router.get("/topics", response_model=list[TopicOut])
def list_topics(db: Session = Depends(get_db)):
    return [_topic_out(db, t) for t in db.query(Topic).order_by(Topic.id.desc()).all()]


@router.get("/topics/{topic_id}", response_model=TopicOut)
def get_topic(topic_id: int, db: Session = Depends(get_db)):
    return _topic_out(db, _topic(db, topic_id))


@router.delete("/topics/{topic_id}")
def delete_topic(topic_id: int, db: Session = Depends(get_db)):
    """Removes the topic, its excerpts and summaries. The papers themselves stay in your library."""
    t = _topic(db, topic_id)
    for model in (TopicPaper, Excerpt, Summary):
        db.query(model).filter(model.topic_id == topic_id).delete()
    db.delete(t)
    db.commit()
    return {"deleted": topic_id}


@router.post("/topics/{topic_id}/search", response_model=DiscoverResponse)
def topic_search(topic_id: int, req: TopicSearchRequest, db: Session = Depends(get_db), embedder=Depends(get_embedder)):
    """Search for papers and store them as this topic's candidate list (replaces the previous list)."""
    t = _topic(db, topic_id)
    q = req.query or t.query or t.name
    cands, warnings = discovery.search_all(q, req.limit, req.sources, get_settings(), embedder)
    stored = []
    for c in cands:
        ex = _find_existing(db, c)
        stored.append(c.to_dict() | {"status": "indexed" if ex else "new", "paper_id": ex.id if ex else None})
    t.candidates = stored
    if req.query:
        t.query = req.query
    db.commit()
    return DiscoverResponse(candidates=[_cand_out(i, d) for i, d in enumerate(stored)], warnings=warnings)


@router.get("/topics/{topic_id}/candidates", response_model=list[CandidateOut])
def topic_candidates(topic_id: int, db: Session = Depends(get_db)):
    return [_cand_out(i, d) for i, d in enumerate(_topic(db, topic_id).candidates or [])]


@router.post("/topics/{topic_id}/import", response_model=list[ImportResult])
def topic_import(topic_id: int, req: ImportRequest, db: Session = Depends(get_db), embedder=Depends(get_embedder),
                 store=Depends(get_store), cache=Depends(get_cache)):
    """Download open-access PDFs for chosen candidates (by index, or the top_n new ones) and index them."""
    t = _topic(db, topic_id)
    s = get_settings()
    cands = [dict(c) for c in (t.candidates or [])]
    if req.indexes is not None:
        bad = [i for i in req.indexes if not 0 <= i < len(cands)]
        if bad:
            raise HTTPException(422, f"Candidate index out of range: {bad}")
        idxs = req.indexes
    else:
        idxs = [i for i, c in enumerate(cands) if c.get("status") == "new"][: req.top_n or 5]
    if not idxs:
        raise HTTPException(409, "No candidates to import - run /topics/{id}/search first")

    results = []
    with discovery.make_client(s) as client:
        for i in idxs:
            c = discovery.Candidate.from_dict(cands[i])
            ex = _find_existing(db, c)
            if ex:
                status, pid, detail = "indexed", ex.id, "already in your library"
            else:
                try:
                    data = discovery.fetch_pdf(c, s, client)
                    paper = ingest_pdf(db, embedder, store, data, safe_filename(c.title, c.arxiv_id), title=c.title,
                                       authors=c.authors, year=c.year, meta={
                                           "source": c.source, "doi": c.doi, "arxiv_id": c.arxiv_id, "url": c.url,
                                           "openalex_id": c.openalex_id, "abstract": c.abstract, "venue": c.venue,
                                           "citation_count": c.citation_count})
                    status, pid, detail = "imported", paper.id, None
                except discovery.DiscoveryError as exc:
                    status, pid, detail = "no_pdf", None, str(exc)[:300]
                except DuplicatePaper as exc:
                    status, pid, detail = "indexed", exc.paper_id, "identical PDF already in your library"
                except IngestError as exc:
                    status, pid, detail = "failed", None, exc.detail
                except Exception as exc:  # noqa: BLE001 - network/vector errors must not abort the batch
                    log.warning("Import failed for '%s': %s", c.title, exc)
                    status, pid, detail = "failed", None, str(exc)[:300]
            if pid:
                _link(db, topic_id, pid, c.relevance)
            cands[i].update(status=status, paper_id=pid)
            results.append(ImportResult(index=i, title=c.title, status=status, paper_id=pid, detail=detail))
    t.candidates = cands
    db.commit()
    cache.clear()
    return results


@router.get("/topics/{topic_id}/papers")
def topic_papers(topic_id: int, db: Session = Depends(get_db)):
    _topic(db, topic_id)
    rel = {r.paper_id: r.relevance for r in db.query(TopicPaper).filter_by(topic_id=topic_id)}
    return [{"paper_id": p.id, "title": p.title, "authors": p.authors, "year": p.year, "source": p.source,
             "relevance": rel.get(p.id, 0.0)} for p in _topic_papers(db, topic_id)]


@router.post("/topics/{topic_id}/papers/{paper_id}", status_code=201)
def attach_paper(topic_id: int, paper_id: int, db: Session = Depends(get_db)):
    """Add a paper you already uploaded to this topic."""
    _topic(db, topic_id)
    if not db.get(Paper, paper_id):
        raise HTTPException(404, f"Paper {paper_id} not found")
    _link(db, topic_id, paper_id, 0.0)
    db.commit()
    return {"topic_id": topic_id, "paper_id": paper_id}


@router.delete("/topics/{topic_id}/papers/{paper_id}")
def detach_paper(topic_id: int, paper_id: int, db: Session = Depends(get_db)):
    _topic(db, topic_id)
    db.query(TopicPaper).filter_by(topic_id=topic_id, paper_id=paper_id).delete()
    db.query(Excerpt).filter_by(topic_id=topic_id, paper_id=paper_id).delete()
    db.commit()
    return {"detached": paper_id}


# ------------------------------------------------------------------ exact sections + notes
@router.post("/topics/{topic_id}/sections", response_model=list[PassageOut])
def topic_sections(topic_id: int, req: SectionsRequest, db: Session = Depends(get_db), embedder=Depends(get_embedder),
                   store=Depends(get_store)):
    """For every paper in the topic, return the section(s) that best match the query (exact text + pages)."""
    t = _topic(db, topic_id)
    q = req.query or t.query or t.name
    ids = [p.id for p in _topic_papers(db, topic_id)]
    if req.paper_ids:
        ids = [i for i in ids if i in req.paper_ids]
    if not ids:
        raise HTTPException(409, "This topic has no papers yet - import some first")
    passages = research.find_passages(store, embedder, q, ids, req.per_paper, req.full_section, req.min_score)
    if req.save:
        _save_excerpts(db, topic_id, q, passages)
        db.commit()
    return [PassageOut(**p) for p in sorted(passages, key=lambda p: -p["score"])]


@router.get("/topics/{topic_id}/excerpts", response_model=list[ExcerptOut])
def list_excerpts(topic_id: int, paper_id: int | None = None, db: Session = Depends(get_db)):
    _topic(db, topic_id)
    q = db.query(Excerpt).filter(Excerpt.topic_id == topic_id)
    if paper_id:
        q = q.filter(Excerpt.paper_id == paper_id)
    return q.order_by(Excerpt.paper_id, Excerpt.id).all()


@router.patch("/excerpts/{excerpt_id}", response_model=ExcerptOut)
def annotate_excerpt(excerpt_id: int, body: NoteIn, db: Session = Depends(get_db)):
    row = db.get(Excerpt, excerpt_id)
    if not row:
        raise HTTPException(404, f"Excerpt {excerpt_id} not found")
    row.note = body.note
    db.commit()
    return row


@router.delete("/excerpts/{excerpt_id}")
def delete_excerpt(excerpt_id: int, db: Session = Depends(get_db)):
    row = db.get(Excerpt, excerpt_id)
    if not row:
        raise HTTPException(404, f"Excerpt {excerpt_id} not found")
    db.delete(row)
    db.commit()
    return {"deleted": excerpt_id}


# ------------------------------------------------------------------ snowballing + references
@router.post("/topics/{topic_id}/expand", response_model=DiscoverResponse)
def topic_expand(topic_id: int, req: ExpandRequest, db: Session = Depends(get_db), embedder=Depends(get_embedder)):
    """Find more papers via the citation graph: what your papers cite, or what cites them (OpenAlex)."""
    t = _topic(db, topic_id)
    s = get_settings()
    papers = [p for p in _topic_papers(db, topic_id) if not req.paper_ids or p.id in req.paper_ids]
    if not papers:
        raise HTTPException(409, "This topic has no papers yet - import some first")
    warnings = []
    with discovery.make_client(s) as client:
        ids = []
        for p in papers:
            if not p.openalex_id:
                p.openalex_id = discovery.resolve_openalex_id(client, s.openalex_api_key, doi=p.doi,
                                                              arxiv_id=p.arxiv_id, title=p.title)
            if p.openalex_id:
                ids.append(p.openalex_id)
            else:
                warnings.append(f"Could not find '{p.title[:60]}' in OpenAlex")
        db.commit()
        if not ids:
            raise HTTPException(502, "None of the topic's papers could be matched in OpenAlex")
        try:
            found = discovery.expand(client, ids, req.direction, req.limit, s.openalex_api_key)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"OpenAlex request failed: {exc}") from exc
    existing = list(t.candidates or [])
    seen = {discovery.norm_title(c["title"]) for c in existing}
    fresh = []
    for c in discovery.rank([c for c in found if discovery.norm_title(c.title) not in seen], t.query or t.name, embedder):
        ex = _find_existing(db, c)
        fresh.append(c.to_dict() | {"status": "indexed" if ex else "new", "paper_id": ex.id if ex else None})
    t.candidates = existing + fresh
    db.commit()
    return DiscoverResponse(candidates=[_cand_out(len(existing) + i, d) for i, d in enumerate(fresh)], warnings=warnings)


def _bibliography(db: Session, topic_id: int) -> list[dict]:
    used = _sections_used(db, topic_id)
    return biblio.assign_keys([_paper_dict(p, used.get(p.id)) for p in _topic_papers(db, topic_id)])


@router.get("/topics/{topic_id}/references")
def topic_references(topic_id: int, format: str = QueryParam("json", pattern="^(json|bibtex|markdown)$"),
                     db: Session = Depends(get_db)):
    """The topic's bibliography: json (with which sections you used), bibtex, or markdown."""
    _topic(db, topic_id)
    refs = _bibliography(db, topic_id)
    if format == "bibtex":
        return PlainTextResponse("\n\n".join(biblio.to_bibtex(r) for r in refs), media_type="text/plain")
    if format == "markdown":
        return PlainTextResponse("\n\n".join(biblio.to_markdown(r, i) for i, r in enumerate(refs, 1)),
                                 media_type="text/markdown")
    return refs


# ------------------------------------------------------------------ summaries + export
@router.post("/topics/{topic_id}/summary", response_model=SummaryOut)
def topic_summary(topic_id: int, req: SummaryRequest, db: Session = Depends(get_db), embedder=Depends(get_embedder),
                  store=Depends(get_store), pipeline=Depends(get_pipeline)):
    """Cited summary of the most relevant papers' matching sections. Uses one LLM call (two+ if very long)."""
    t = _topic(db, topic_id)
    focus = req.focus or t.query or t.name
    papers = {p.id: p for p in _topic_papers(db, topic_id)}
    if not papers:
        raise HTTPException(409, "This topic has no papers yet - import some first")
    passages = research.find_passages(store, embedder, focus, list(papers), req.per_paper, full_section=False,
                                      min_score=get_settings().min_score)
    best: dict[int, float] = {}
    for ps in passages:
        best[ps["paper_id"]] = max(best.get(ps["paper_id"], 0), ps["score"])
    chosen = sorted(best, key=lambda pid: -best[pid])[: req.max_papers]
    if not chosen:
        raise HTTPException(422, "No sections matched this focus - try broader wording or import more papers")
    sources = [{"title": papers[pid].title, "year": papers[pid].year,
                "passages": [p for p in passages if p["paper_id"] == pid]} for pid in chosen]
    body, provider = research.summarize(pipeline.answer_router, focus, sources, req.style, get_settings().llm_max_tokens)

    used = {pid: [f"{p['section']} p.{p['page_start']}" + (f"-{p['page_end']}" if p['page_end'] != p['page_start'] else "")
                  for p in passages if p["paper_id"] == pid] for pid in chosen}
    refs = [biblio.to_markdown(_paper_dict(papers[pid], used[pid]), i) for i, pid in enumerate(chosen, 1)]
    markdown = f"# {t.name}: {focus}\n\n{body.strip()}\n\n## References\n\n" + "\n\n".join(refs)
    row = Summary(topic_id=topic_id, focus=focus, markdown=markdown, paper_ids=chosen, llm_provider=provider)
    db.add(row)
    _save_excerpts(db, topic_id, focus, [p for p in passages if p["paper_id"] in chosen], overwrite=False)
    db.commit()
    return row


@router.get("/topics/{topic_id}/summaries", response_model=list[SummaryOut])
def list_summaries(topic_id: int, db: Session = Depends(get_db)):
    _topic(db, topic_id)
    return db.query(Summary).filter(Summary.topic_id == topic_id).order_by(Summary.id.desc()).all()


@router.get("/topics/{topic_id}/export")
def export_topic(topic_id: int, db: Session = Depends(get_db)):
    """One markdown document: latest summary, your saved excerpts per paper, references and BibTeX."""
    t = _topic(db, topic_id)
    refs = _bibliography(db, topic_id)
    summary = db.query(Summary).filter(Summary.topic_id == topic_id).order_by(Summary.id.desc()).first()
    lines = [f"# {t.name}", f"_Query: {t.query}_ · exported {datetime.now(timezone.utc):%Y-%m-%d}", ""]
    if summary:
        lines += ["## Summary", "", summary.markdown.split("\n", 2)[-1].strip(), ""]
    lines += ["## Saved excerpts", ""]
    for r in refs:
        rows = db.query(Excerpt).filter_by(topic_id=topic_id, paper_id=r["id"]).order_by(Excerpt.id).all()
        if not rows:
            continue
        lines += [f"### {r['title']} ({r['year'] or 'n.d.'})", ""]
        for e in rows:
            pages = e.page_start if e.page_start == e.page_end else f"{e.page_start}-{e.page_end}"
            lines += [f"**{e.section}, p. {pages}** (match {e.score:.2f})", "", e.text, ""]
            if e.note:
                lines += [f"> Note: {e.note}", ""]
    lines += ["## References", ""] + [biblio.to_markdown(r, i) + "\n" for i, r in enumerate(refs, 1)]
    lines += ["## BibTeX", "", "```bibtex", "\n\n".join(biblio.to_bibtex(r) for r in refs), "```"]
    return PlainTextResponse("\n".join(lines), media_type="text/markdown")
