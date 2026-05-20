"""
    Ingestion Module — Incremental Sync
    
    Tracks file modification times in a JSON state file to determine
    what needs to be (re-)ingested without wiping the entire ChromaDB.

    Functions:
        sync_all_documents()      — scan docs/ and sync changed/new files
        ingest_single_file(path)  — embed and add a specific file (used by watcher)
        remove_file(path)         — delete a file's chunks from ChromaDB
"""

import json
import os
import sys
from pathlib import Path
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import log

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DOCS_DIR   = BASE_DIR / "docs"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
STATE_FILE = str(BASE_DIR / "ingest_state.json")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".json", ".xml", ".html", ".md", ".txt", ".docx"}

# ── Shared resources (lazy-loaded so this module is importable without side-effects) ──
_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings(),
        )
    return _vectorstore


def inject_vectorstore(vs: Chroma):
    """Allow rag_server.py to share its already-loaded vectorstore instance."""
    global _vectorstore
    _vectorstore = vs


# ── State helpers ──────────────────────────────────────────────────────────────

def _load_state() -> dict:
    """Return the persisted file-state dict: {abs_path: mtime}."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_state(state: dict):
    """Persist the file-state dict."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ── Metadata ──────────────────────────────────────────────────────────────────

def extract_metadata(file_path: Path) -> dict:
    """Derive category / author from folder structure under docs/."""
    try:
        rel = file_path.relative_to(DOCS_DIR)
        parts = rel.parts
        meta = {"path": str(file_path)}
        if parts:
            meta["category"] = parts[0]
            if parts[0] == "ctf-writeups" and len(parts) >= 2:
                meta["author"] = parts[1]
        return meta
    except Exception as e:
        log.warn(f"Metadata extraction failed for {file_path}: {e}")
        return {"path": str(file_path), "category": "unknown"}


# ── Core ingestion ─────────────────────────────────────────────────────────────

def _load_and_chunk(file_path: Path) -> list:
    """Load a single file and split it into chunks with metadata."""
    loader = UnstructuredLoader(str(file_path))
    docs = loader.load()
    meta = extract_metadata(file_path)
    for doc in docs:
        doc.metadata.update(meta)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    return splitter.split_documents(docs)


def _remove_file_chunks(file_path: Path):
    """Delete all ChromaDB chunks whose 'path' metadata matches this file."""
    vs = get_vectorstore()
    abs_path = str(file_path)
    try:
        results = vs.get(where={"path": abs_path})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            vs.delete(ids=ids_to_delete)
            log.info(f"Deleted {len(ids_to_delete)} old chunks for: {file_path.name}")
    except Exception as e:
        log.warn(f"Could not remove old chunks for {file_path.name}: {e}")


def ingest_single_file(file_path: Path | str, state: dict | None = None) -> bool:
    """
    Embed and add a single file to ChromaDB.
    Removes existing chunks for that file first to prevent duplicates.
    Optionally updates the provided state dict in-place (caller saves it).
    Returns True on success.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        log.warn(f"File not found, skipping: {file_path}")
        return False
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False

    log.info(f"Ingesting: {file_path.relative_to(BASE_DIR)}")
    try:
        _remove_file_chunks(file_path)
        chunks = _load_and_chunk(file_path)
        if not chunks:
            log.warn(f"No content extracted from {file_path.name}")
            return False

        vs = get_vectorstore()
        vs.add_documents(chunks)
        log.info(f"  ↳ {len(chunks)} chunks added for {file_path.name}")

        if state is not None:
            state[str(file_path)] = file_path.stat().st_mtime
        return True

    except Exception as e:
        log.error(f"Failed to ingest {file_path.name}: {e}")
        return False


def remove_file(file_path: Path | str):
    """Remove all ChromaDB chunks for a deleted file and update state."""
    file_path = Path(file_path)
    _remove_file_chunks(file_path)
    state = _load_state()
    state.pop(str(file_path), None)
    _save_state(state)
    log.info(f"Removed from index: {file_path.name}")


# ── Full directory sync ────────────────────────────────────────────────────────

def sync_all_documents():
    """
    Incrementally synchronise the ChromaDB with the docs/ directory.
    - New files are ingested.
    - Modified files (by mtime) have their old chunks replaced.
    - Deleted files have their chunks removed.
    - Unchanged files are skipped.
    """
    if not DOCS_DIR.exists():
        log.error(f"docs/ directory not found at {DOCS_DIR}")
        return

    # Ensure ChromaDB dir exists (first-time setup)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    state = _load_state()
    disk_files: dict[str, float] = {}

    # Collect all supported files on disk
    for file_path in DOCS_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            disk_files[str(file_path)] = file_path.stat().st_mtime

    new_count = modified_count = deleted_count = skipped_count = 0

    # Detect deletions (in state but not on disk)
    for abs_path in list(state.keys()):
        if abs_path not in disk_files:
            _remove_file_chunks(Path(abs_path))
            del state[abs_path]
            deleted_count += 1
            log.info(f"Removed deleted file from index: {Path(abs_path).name}")

    # Detect new and modified files
    for abs_path, mtime in disk_files.items():
        file_path = Path(abs_path)
        prev_mtime = state.get(abs_path)

        if prev_mtime is None:
            # New file
            if ingest_single_file(file_path, state):
                new_count += 1
        elif abs(mtime - prev_mtime) > 0.01:
            # Modified file (mtime changed)
            if ingest_single_file(file_path, state):
                modified_count += 1
        else:
            skipped_count += 1

    _save_state(state)

    summary = (
        f"Sync complete — "
        f"{new_count} new, {modified_count} updated, "
        f"{deleted_count} removed, {skipped_count} unchanged"
    )
    log.info(summary)

    # Write to a dedicated sync log file so the user can track completion
    from datetime import datetime
    sync_log_path = BASE_DIR / "sync.log"
    try:
        with open(sync_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {summary}\n")
    except Exception as e:
        log.warn(f"Could not write to sync log: {e}")


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.banner("Ingestion Sync")
    log.info("Loading embedding model...")
    with log.Timer() as t:
        get_embeddings()
    log.info(f"Embedding model loaded ({t.elapsed_ms:.0f}ms)")

    log.info("Connecting to ChromaDB...")
    with log.Timer() as t:
        get_vectorstore()
    log.info(f"ChromaDB ready ({t.elapsed_ms:.0f}ms)")

    sync_all_documents()
