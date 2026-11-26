"""Local sentence-transformers embeddings (lazy-loaded, thread-safe)."""
import logging
import threading

log = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info("Loading embedding model %s", self.model_name)
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
