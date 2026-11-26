"""FastAPI entrypoint.  Run:  uvicorn src.main:app --reload"""
import logging
import logging.handlers
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from src.api.deps import get_embedder, get_store
from src.api.research import router as research_router
from src.api.routes import router
from src.config import get_settings
from src.models import db as database
from src.services.llm import LLMError, configured_providers
from src.services.qdrant_client import VectorStoreError

log = logging.getLogger("rpa")


def setup_logging(level: str, log_file: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3))
    logging.basicConfig(level=level.upper(), handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(s.log_level, s.log_file)
    database.init_db(s.database_url)
    if s.warmup_models:
        try:
            get_embedder().dim  # loads the model once at startup
            get_store()
        except Exception as exc:  # keep the API up; /health will report the problem
            log.error("Startup warm-up failed: %s", exc)
    log.info("Started. LLM providers: %s", configured_providers(s) or "NONE CONFIGURED")
    yield


app = FastAPI(title="Research Paper Assistant", version="1.0.0", lifespan=lifespan,
              description="RAG service for querying academic papers with citations.")
app.include_router(router)
app.include_router(research_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code,
             (time.perf_counter() - t0) * 1000)
    return response


@app.exception_handler(VectorStoreError)
async def vector_error(_: Request, exc: VectorStoreError):
    log.error("Vector store error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Vector database unavailable. Is Qdrant running?"})


@app.exception_handler(LLMError)
async def llm_error(_: Request, exc: LLMError):
    log.error("LLM error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": f"Language model unavailable: {exc}"})


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health():
    s = get_settings()
    checks = {}
    try:
        with database.get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"
    try:
        checks["qdrant"] = "ok" if get_store().healthy() else "unreachable"
    except Exception:
        checks["qdrant"] = "unreachable"
    checks["llm_providers"] = configured_providers(s)
    ok = checks["database"] == "ok" and checks["qdrant"] == "ok" and bool(checks["llm_providers"])
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "degraded", **checks})
