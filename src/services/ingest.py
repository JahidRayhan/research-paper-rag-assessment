"""Shared PDF ingestion used by manual uploads and by literature-discovery imports."""
import hashlib
import logging
import os
import re
import time

from sqlalchemy.orm import Session

from src.config import get_settings
from src.models.db import Paper
from src.services.pdf_processor import PDFProcessingError, chunk_blocks, parse_pdf
from src.services.qdrant_client import VectorStoreError

log = logging.getLogger(__name__)
META_FIELDS = {"source", "doi", "arxiv_id", "openalex_id", "url", "abstract", "venue", "citation_count"}


class IngestError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


class DuplicatePaper(IngestError):
    def __init__(self, paper_id: int):
        super().__init__(409, "This paper is already indexed")
        self.paper_id = paper_id


def safe_filename(title: str, arxiv_id: str | None = None) -> str:
    if arxiv_id:
        return "arxiv_" + re.sub(r"[^A-Za-z0-9.]+", "_", arxiv_id) + ".pdf"
    return (re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:60] or "paper") + ".pdf"


def ingest_pdf(db: Session, embedder, store, data: bytes, filename: str, *, title: str | None = None,
               authors: list[str] | None = None, year: int | None = None, meta: dict | None = None) -> Paper:
    s = get_settings()
    t0 = time.perf_counter()
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise IngestError(413, f"File larger than {s.max_upload_mb} MB")
    if not data.startswith(b"%PDF"):
        raise IngestError(415, "File does not look like a valid PDF")
    digest = hashlib.sha256(data).hexdigest()
    existing = db.query(Paper).filter(Paper.file_hash == digest).first()
    if existing:
        raise DuplicatePaper(existing.id)

    os.makedirs(s.upload_dir, exist_ok=True)
    stored = os.path.join(s.upload_dir, f"{digest[:16]}.pdf")
    with open(stored, "wb") as f:
        f.write(data)

    paper_id = None
    try:
        parsed = parse_pdf(stored)
        skip = () if s.index_references else ("References",)
        chunks = chunk_blocks(parsed.blocks, s.chunk_target_words, s.chunk_max_words, s.chunk_overlap_sentences, skip)
        if not chunks:
            raise PDFProcessingError("No indexable text found in PDF")
        extra = {k: v for k, v in (meta or {}).items() if k in META_FIELDS}
        paper = Paper(title=(title or parsed.title).strip()[:500], authors=authors or parsed.authors,
                      year=year or parsed.year, filename=filename, file_hash=digest, stored_path=stored,
                      num_pages=parsed.num_pages, num_chunks=len(chunks), sections=parsed.sections, **extra)
        db.add(paper)
        db.flush()
        paper_id = paper.id
        # Prefixing title + section gives each chunk document-level context at embedding time.
        vectors = embedder.embed_documents([f"{paper.title} | {c.section}\n{c.text}" for c in chunks])
        payloads = [{"paper_title": paper.title, "filename": filename, "section": c.section, "chunk_index": c.index,
                     "page_start": c.page_start, "page_end": c.page_end, "text": c.text} for c in chunks]
        store.upsert_chunks(paper_id, vectors, payloads)
        db.commit()
    except Exception as exc:
        db.rollback()
        if paper_id is not None:
            try:
                store.delete_paper(paper_id)
            except VectorStoreError:
                log.warning("Could not clean up vectors for failed paper %s", paper_id)
        if os.path.exists(stored):
            os.remove(stored)
        if isinstance(exc, PDFProcessingError):
            raise IngestError(422, str(exc)) from exc
        if isinstance(exc, VectorStoreError):
            raise
        log.exception("Indexing failed for %s", filename)
        raise IngestError(500, "Failed to index paper") from exc
    log.info("Indexed '%s' (id=%s, %d chunks) in %.1fs", paper.title, paper.id, len(chunks), time.perf_counter() - t0)
    return paper
