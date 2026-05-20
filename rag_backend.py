"""
    RAG Backend Server
    
    Runs as a standalone HTTP API server (FastAPI + uvicorn).
    Owns all heavy resources: ChromaDB, HuggingFace embeddings, CrossEncoder reranker,
    incremental document ingestion, and the live filesystem watcher.

    The MCP server (rag_server.py) is a thin client that calls this backend.

    Usage:
        python rag_backend.py
        python rag_backend.py --host 127.0.0.1 --port 8000

    Endpoints:
        GET  /health        — readiness probe (returns {"status": "ready"} once loaded)
        POST /search        — run the RAG pipeline
        GET  /categories    — list available doc categories
"""

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import log
import ingest as ingestion

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DOCS_DIR     = BASE_DIR / "docs"
CHROMA_DIR   = str(BASE_DIR / "chroma_db")
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVE_K = 20   # candidates fetched from vector store
RETURN_K   = 5    # top-N kept after reranking

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".json", ".xml", ".html", ".md", ".txt", ".docx"}

# ── Global state ───────────────────────────────────────────────────────────────
_ready        = False
_vectorstore  = None
_reranker     = None
_categories   = []
_watcher      = None


# ── Live Document Watcher ──────────────────────────────────────────────────────

class DocWatcher(FileSystemEventHandler):
    """Monitors docs/ for file system events and triggers live ingestion."""

    def _is_supported(self, path: str) -> bool:
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS

    def on_created(self, event):
        if not event.is_directory and self._is_supported(event.src_path):
            log.info(f"[Watcher] New file detected: {Path(event.src_path).name}")
            ingestion.ingest_single_file(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory and self._is_supported(event.src_path):
            log.info(f"[Watcher] File modified: {Path(event.src_path).name}")
            ingestion.ingest_single_file(Path(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory and self._is_supported(event.src_path):
            log.info(f"[Watcher] File removed: {Path(event.src_path).name}")
            ingestion.remove_file(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            if self._is_supported(event.src_path):
                log.info(f"[Watcher] File moved from: {Path(event.src_path).name}")
                ingestion.remove_file(Path(event.src_path))
            if self._is_supported(event.dest_path):
                log.info(f"[Watcher] File moved to: {Path(event.dest_path).name}")
                ingestion.ingest_single_file(Path(event.dest_path))


def _start_watcher() -> Observer:   #type: ignore
    observer = Observer()
    observer.schedule(DocWatcher(), str(DOCS_DIR), recursive=True)
    observer.start()
    log.info(f"[Watcher] Monitoring {DOCS_DIR} for changes...")
    return observer


# ── FastAPI Lifespan ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources at startup, then hand off to the app."""
    global _ready, _vectorstore, _reranker, _categories, _watcher

    log.banner("RAG Backend Server", f"ChromaDB: {CHROMA_DIR}")

    # 1. Load embedding model
    log.info("Loading embedding model...")
    with log.Timer() as t:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    log.info(f"Embedding model loaded ({t.elapsed_ms:.0f}ms)")

    # 2. Connect to ChromaDB
    os.makedirs(CHROMA_DIR, exist_ok=True)
    log.info("Connecting to ChromaDB...")
    with log.Timer() as t:
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
    log.info(f"ChromaDB ready ({t.elapsed_ms:.0f}ms)")

    # Share instances with the ingestion module to avoid double-loading
    ingestion.inject_vectorstore(_vectorstore)
    ingestion._embeddings = embeddings

    # 3. Incremental sync (blocking — intentional, so you see it complete before use)
    log.info("Running incremental document sync...")
    with log.Timer() as t:
        ingestion.sync_all_documents()
    log.info(f"Startup sync complete ({t.elapsed_ms:.0f}ms)")

    # 4. Load CrossEncoder reranker
    log.info("Loading CrossEncoder reranker...")
    with log.Timer() as t:
        _reranker = CrossEncoder(RERANK_MODEL)
    log.info(f"CrossEncoder loaded ({t.elapsed_ms:.0f}ms)")

    # 5. Auto-detect categories and start watcher
    _categories = sorted(
        [d.name for d in DOCS_DIR.iterdir() if d.is_dir()]
    ) if DOCS_DIR.exists() else []
    category_list = ", ".join(f"'{c}'" for c in _categories) or "(none found)"
    log.info(f"Available categories: {category_list}")

    _watcher = _start_watcher()

    # 6. Mark ready
    _ready = True
    log.info(" [*] RAG Backend is READY — accepting search requests")

    yield  # ← Server runs here

    # Shutdown
    log.info("Shutting down RAG Backend...")
    if _watcher:
        _watcher.stop()
        _watcher.join()
    log.info("RAG Backend stopped.")


# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Backend", lifespan=lifespan)


# ── Request / Response Models ──────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    category: str | None = None


class SearchResponse(BaseModel):
    results: str
    elapsed_ms: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    if _ready:
        return {"status": "ready", "categories": _categories}
    return JSONResponse({"status": "loading"}, status_code=503)


@app.get("/categories")
def categories():
    return {"categories": _categories}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not _ready:
        return JSONResponse({"error": "Backend is still initialising. Please retry shortly."}, status_code=503)

    log.request("search", {"query": req.query, "category": req.category})

    try:
        # ── Stage 1: Broad vector retrieval ──────────────────────────────────
        search_kwargs: dict = {"k": RETRIEVE_K}
        if req.category:
            search_kwargs["filter"] = {"category": req.category}

        with log.Timer() as t_retrieve:
            candidates = _vectorstore.similarity_search(req.query, **search_kwargs)

        if not candidates:
            summary = "No relevant information found in the knowledge base."
            log.response("ChromaDB", summary, t_retrieve.elapsed_ms)
            return SearchResponse(results=summary, elapsed_ms=t_retrieve.elapsed_ms)

        log.info(f"Retrieved {len(candidates)} candidates ({t_retrieve.elapsed_ms:.0f}ms)")

        # ── Stage 2: Reranking with CrossEncoder ──────────────────────────────
        pairs = [(req.query, doc.page_content) for doc in candidates]
        with log.Timer() as t_rerank:
            scores = _reranker.predict(pairs)
        log.info(f"Reranked {len(scores)} candidates ({t_rerank.elapsed_ms:.0f}ms)")

        ranked    = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top       = ranked[:RETURN_K]

        # ── Stage 3: Format output ────────────────────────────────────────────
        formatted = []
        for score, doc in top:
            source_cat = doc.metadata.get("category", "unknown")
            path       = doc.metadata.get("path", "unknown")
            content    = doc.page_content.replace("\n", " ").strip()
            formatted.append(
                f"[Source: {source_cat}] (Path: {path}) [Score: {score:.3f}]\n{content}"
            )

        output     = "\n\n---\n\n".join(formatted)
        total_ms   = t_retrieve.elapsed_ms + t_rerank.elapsed_ms
        log.response("Pipeline", f"Returned {len(top)} reranked results", total_ms)
        return SearchResponse(results=output, elapsed_ms=total_ms)

    except Exception as e:
        log.error(f"Search pipeline failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Backend HTTP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",   # suppress uvicorn's own chatter; we use our log module
    )
