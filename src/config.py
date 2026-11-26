"""Central configuration. Everything comes from environment variables / .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"
    log_file: str | None = "logs/app.log"
    api_key: str | None = None  # if set, clients must send X-API-Key
    warmup_models: bool = True

    database_url: str = "sqlite:///./data/app.db"
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 50

    qdrant_url: str = "http://localhost:6333"  # ":memory:" works for quick dev
    qdrant_api_key: str | None = None
    qdrant_collection: str = "papers"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_target_words: int = 200
    chunk_max_words: int = 300
    chunk_overlap_sentences: int = 1
    index_references: bool = False

    # LLM routing: comma separated, first = primary, rest = fallbacks
    answer_providers: str = "claude,gemini"
    rewrite_providers: str = "gemini,claude"
    rewrite_enabled: bool = False
    anthropic_api_key: str | None = None
    claude_model: str = "claude-sonnet-5-5"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"  # verify in AI Studio: model IDs retire
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    llm_timeout_s: int = 60
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1

    # literature discovery (all free). OpenAlex needs a free key: https://openalex.org/settings/api
    openalex_api_key: str | None = None
    contact_email: str | None = None      # goes into the User-Agent (polite to arXiv/Unpaywall)
    unpaywall_email: str | None = None    # optional: enables Unpaywall as an open-access PDF fallback
    discovery_timeout_s: int = 30
    arxiv_interval_s: float = 3.1         # arXiv asks for max 1 request / 3 s

    min_score: float = 0.20  # drop chunks below this cosine similarity
    cache_ttl_s: int = 300


@lru_cache
def get_settings() -> Settings:
    return Settings()
