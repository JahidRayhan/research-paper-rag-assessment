import hashlib
import math
import re

import pymupdf as fitz
import pytest


def make_pdf(path, title, authors, sections, meta_title=None):
    """Build a small paper-like PDF. sections = [(heading, [paragraph, ...]), ...]"""
    doc = fitz.open()
    state = {"page": doc.new_page(), "y": 60}

    def box(text, size, h):
        if state["y"] + h > 760:
            state["page"], state["y"] = doc.new_page(), 60
        rc = state["page"].insert_textbox(fitz.Rect(60, state["y"], 550, state["y"] + h), text, fontsize=size)
        assert rc >= 0, f"text did not fit: {text[:30]}"
        state["y"] += h + 8

    box(title, 18, 50)
    box(authors, 11, 24)
    box("Journal of Tests 2017", 9, 16)
    for heading, paras in sections:
        box(heading, 12, 22)
        for p in paras:
            box(p, 10, 150)
    if meta_title:
        doc.set_metadata({"title": meta_title})
    doc.save(path)
    doc.close()


def paragraph(topic: str, n: int = 6) -> str:
    return " ".join(f"Sentence {i} about {topic} explains the {topic} approach in considerable detail." for i in range(n))


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download needed)."""
    dim = 256

    def _vec(self, text):
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z]+", text.lower()):
            if len(tok) > 3:
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts, batch_size=32):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def pdf_factory(tmp_path):
    def _make(name="paper.pdf", **kw):
        p = tmp_path / name
        make_pdf(str(p), **kw)
        return str(p)
    return _make


# ---------------------------------------------------------------- shared API fixture
from fastapi.testclient import TestClient  # noqa: E402

from src.api import deps  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.services.cache import TTLCache  # noqa: E402
from src.services.llm import LLMProvider, LLMRouter  # noqa: E402
from src.services.qdrant_client import VectorStore  # noqa: E402
from src.services.rag_pipeline import RAGPipeline  # noqa: E402


class EchoLLM(LLMProvider):
    name = "fake"
    def generate(self, system, prompt, max_tokens, temperature):
        return "Self-attention relates all positions of a sequence [1][7]."


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with in-memory Qdrant, fake embedder and fake LLM (no network, no model downloads)."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "up"))
    monkeypatch.setenv("WARMUP_MODELS", "false")
    monkeypatch.setenv("LOG_FILE", "")
    monkeypatch.setenv("MIN_SCORE", "0.05")
    get_settings.cache_clear()
    from src.main import app

    emb, store, cache = FakeEmbedder(), VectorStore(":memory:", "t", FakeEmbedder.dim), TTLCache(60)
    store.ensure_collection()
    pipeline = RAGPipeline(get_settings(), emb, store, LLMRouter([EchoLLM()]))
    app.dependency_overrides = {deps.get_embedder: lambda: emb, deps.get_store: lambda: store,
                                deps.get_pipeline: lambda: pipeline, deps.get_cache: lambda: cache}
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = {}
    get_settings.cache_clear()
