"""Thin wrapper around Qdrant (cosine similarity, payload filter on paper_id)."""
import logging
import uuid

from qdrant_client import QdrantClient  # the third-party library (absolute import)
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchAny, MatchValue,
                                  PayloadSchemaType, PointStruct, VectorParams)

log = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    pass


class VectorStore:
    def __init__(self, url: str, collection: str, dim: int, api_key: str | None = None):
        self.collection, self.dim = collection, dim
        self.client = (QdrantClient(location=":memory:") if url == ":memory:"
                       else QdrantClient(url=url, api_key=api_key, timeout=30))

    def ensure_collection(self) -> None:
        try:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    self.collection, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE))
                self.client.create_payload_index(self.collection, "paper_id", PayloadSchemaType.INTEGER)
                self.client.create_payload_index(self.collection, "section", PayloadSchemaType.KEYWORD)
                log.info("Created Qdrant collection %s (dim=%d)", self.collection, self.dim)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant unavailable: {exc}") from exc

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def upsert_chunks(self, paper_id: int, vectors: list[list[float]], payloads: list[dict]) -> None:
        points = [PointStruct(id=str(uuid.uuid4()), vector=v, payload={**p, "paper_id": paper_id})
                  for v, p in zip(vectors, payloads)]
        try:
            for i in range(0, len(points), 64):
                self.client.upsert(self.collection, points=points[i:i + 64], wait=True)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

    def search(self, vector: list[float], limit: int, paper_ids: list[int] | None = None) -> list[dict]:
        flt = Filter(must=[FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))]) if paper_ids else None
        try:
            res = self.client.query_points(self.collection, query=vector, limit=limit,
                                           query_filter=flt, with_payload=True)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc
        return [{"score": float(p.score), **(p.payload or {})} for p in res.points]

    def get_section(self, paper_id: int, section: str) -> list[dict]:
        """All chunks of one section of one paper, in document order."""
        flt = Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id)),
                           FieldCondition(key="section", match=MatchValue(value=section))])
        out, offset = [], None
        try:
            while True:
                points, offset = self.client.scroll(self.collection, scroll_filter=flt, limit=256, offset=offset,
                                                    with_payload=True, with_vectors=False)
                out += [p.payload or {} for p in points]
                if offset is None:
                    break
        except Exception as exc:
            raise VectorStoreError(f"Qdrant scroll failed: {exc}") from exc
        return sorted(out, key=lambda p: p.get("chunk_index", 0))

    def delete_paper(self, paper_id: int) -> None:
        try:
            from qdrant_client.models import FilterSelector

            self.client.delete(self.collection, points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))])))
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc
