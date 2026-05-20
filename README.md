# Icebreaker MCP Tools

A modular Model Context Protocol (MCP) server toolkit for cybersecurity research and knowledge retrieval. This project includes MCP servers for web search via SearXNG and local knowledge base queries powered by RAG.

## Overview

### Components

- **RAG Backend**: A FastAPI server that manages embeddings, vector storage, and intelligent document reranking
- **RAG MCP Server**: An MCP proxy that exposes the backend as a tool for language models
- **SearXNG MCP Server**: An MCP wrapper for web search functionality
- **Ingestion Pipeline**: Automatic document processing and synchronization
- **Logging Utilities**: Structured console output for debugging

## Scripts

### 1. rag_backend.py


The core FastAPI server that handles all heavy lifting for the local knowledge base system.

**Features:**
- Manages ChromaDB vector store for document embeddings
- Loads and runs HuggingFace embedding model (sentence-transformers/all-MiniLM-L6-v2)
- Implements a CrossEncoder reranker for relevance scoring (ms-marco-MiniLM-L-6-v2)
- Monitors the docs/ folder for file changes with live ingestion
- Provides HTTP endpoints for search and document management
- Runs on http://127.0.0.1:8000 by default

**Usage:**
```bash
python rag_backend.py                                    # default: localhost:8000
python rag_backend.py --host 127.0.0.1 --port 8000
```

**Endpoints:**
- `GET /health` - Health/readiness check
- `POST /search` - Execute RAG search pipeline
- `GET /categories` - List available document categories

---

### 2. rag_server.py

A lightweight MCP proxy server that exposes the RAG backend as an MCP tool for language models.

**Features:**
- Connects to rag_backend.py via HTTP
- Implements a single MCP tool: search_local_knowledge()
- Gracefully handles backend offline scenarios
- Supports both stdio and SSE transport modes
- Caches available categories on startup

**Usage:**
```bash
python rag_server.py                                     # stdio mode (default, for LM Studio)
python rag_server.py --transport sse --port 8002        # SSE mode for remote connections
```

**MCP Tool:**
- `search_local_knowledge(query, category=None)` - Search the knowledge base
  - Retrieves top-20 candidates, reranks them, and returns top-5 results
  - Category parameter is optional and filters by document folder

---

### 3. ingest.py

Handles incremental document ingestion into ChromaDB with smart change detection.

**Features:**
- Tracks file modification times in a JSON state file
- Only processes changed or new documents (skips unchanged files)
- Supports multiple file formats: PDF, CSV, JSON, XML, HTML, MD, TXT, DOCX
- Extracts metadata (category, author) from folder structure
- Uses RecursiveCharacterTextSplitter for intelligent text chunking
- HuggingFace embeddings for document vectorization

**Core Functions:**
- `sync_all_documents()` - Scan and sync all changed/new files in docs/
- `ingest_single_file(path)` - Embed and add a specific file (used by filesystem watcher)
- `remove_file(path)` - Delete a file's chunks from ChromaDB

---

### 4. server.py

An MCP server for web search functionality via a local SearXNG instance.

**Features:**
- Integrates with local SearXNG search engine (requires Docker container running)
- Supports filtered search by categories (general, images, news, videos, it, science, etc.)
- Pagination support for browsing through results
- Configurable result limiting
- Graceful error handling for offline SearXNG

**Usage:**
```bash
python server.py --transport sse --host <host-ip> --port 8001    # HTTP SSE mode
python server.py --transport stdio                               # Stdio mode
```

**MCP Tool:**
- `search_web(query, categories="general", page=1, max_results=10)` - Search the web
  - Accepts category filters for specialized searches
  - Returns up to 20 results per page with snippets

---

### 5. log.py

A stylized console logging utility with ANSI color codes for structured debug output.

**Features:**
- Color-coded output: green for requests, white for responses, cyan for info, yellow for warnings, red for errors
- Box-drawing characters for visual organization
- Timestamp logging with millisecond precision
- Performance timing via context manager
- Writes to stderr to avoid polluting MCP stdio transport

**Functions:**
- `banner(server_name, detail)` - Print startup banner
- `info(msg)` - Log informational message
- `warn(msg)` - Log warning
- `error(msg)` - Log error
- `request(tool_name, params)` - Log incoming tool call
- `response(tool_name, summary, elapsed_ms)` - Log service response
- `Timer()` - Context manager to measure elapsed time

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Language Model / MCP Client                                │
└────────────┬──────────────────────────────┬─────────────────┘
             │                              │
      ┌──────▼────────┐            ┌────────▼──────┐
      │ rag_server.py │            │ server.py     │
      │ (MCP Proxy)   │            │ (SearXNG MCP) │
      └──────┬────────┘            └────────┬──────┘
             │                              │
      ┌──────▼──────────────────┐    ┌──────▼─────────┐
      │ rag_backend.py (FastAPI)│    │ SearXNG Docker │
      │ :8000                   │    │ Container      │
      └──────┬──────────────────┘    └────────────────┘
             │
      ┌──────▼──────────────────┐
      │ ChromaDB + Embeddings   │
      │ CrossEncoder Reranker   │
      └──────┬──────────────────┘
             │
      ┌──────▼──────────────────┐
      │ docs/ folder            │
      │ (watched for changes)   │
      └─────────────────────────┘
```

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the RAG backend (handles document indexing and search):
```bash
python rag_backend.py
```
Wait for the message indicating the backend is ready.

3. In a new terminal, start the RAG MCP server:
```bash
python rag_server.py
```

4. (Optional) For web search, ensure SearXNG is running via Docker:
```bash
docker-compose up -d
```
Then start the search MCP server:
```bash
python server.py --transport stdio
```

## Configuration

All configuration is defined at the top of each script:

- **rag_backend.py / ingest.py:**
  - `EMBED_MODEL` - HuggingFace embedding model
  - `RERANK_MODEL` - CrossEncoder model for reranking
  - `RETRIEVE_K` - Number of candidates to retrieve (default: 20)
  - `RETURN_K` - Number of results to return after reranking (default: 5)
  - `CHROMA_DIR` - Location of ChromaDB storage
  - `DOCS_DIR` - Root directory for documents to ingest

- **rag_server.py:**
  - `BACKEND_URL` - HTTP endpoint for rag_backend.py
  - `RETRIEVE_K`, `RETURN_K` - Same as backend

- **server.py:**
  - `SEARXNG_BASE_URL` - URL of local SearXNG instance

## Supported Document Formats

The ingestion pipeline supports: PDF, CSV, JSON, XML, HTML, Markdown, TXT, DOCX

## State Management

The system maintains incremental state:
- `ingest_state.json` - Tracks file modification times to determine what needs re-ingestion
- `chroma_db/` - Vector database directory with persisted embeddings

Deleting these files will cause a full re-ingest on the next run.
