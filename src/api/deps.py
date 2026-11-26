"""FastAPI dependencies / lazily created singletons (override these in tests)."""
import hmac
from functools import lru_cache

from fastapi import Header, HTTPException

from src.config import get_settings
from src.models.db import get_db  # noqa: F401  (re-exported)
from src.services.cache import TTLCache
from src.services.embedding_service import EmbeddingService
from src.services.llm import build_router
from src.services.qdrant_client import VectorStore
from src.services.rag_pipeline import RAGPipeline


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    key = get_settings().api_key
    if key and not hmac.compare_digest(x_api_key or "", key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


@lru_cache
def get_embedder() -> EmbeddingService:
    return EmbeddingService(get_settings().embedding_model)


@lru_cache
def get_store() -> VectorStore:
    s = get_settings()
    store = VectorStore(s.qdrant_url, s.qdrant_collection, get_embedder().dim, s.qdrant_api_key)
    store.ensure_collection()
    return store


@lru_cache
def get_pipeline() -> RAGPipeline:
    s = get_settings()
    rewrite = build_router(s.rewrite_providers, s) if s.rewrite_enabled else None
    return RAGPipeline(s, get_embedder(), get_store(), build_router(s.answer_providers, s), rewrite)


@lru_cache
def get_cache() -> TTLCache:
    return TTLCache(get_settings().cache_ttl_s)
