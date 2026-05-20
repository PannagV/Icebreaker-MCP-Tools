"""
    LocalKnowledgeServer — MCP Proxy

    A lightweight FastMCP server.  All heavy RAG work (embeddings, ChromaDB,
    CrossEncoder, ingestion) lives in rag_backend.py.  This file only handles
    the MCP protocol and forwards search requests to the backend over HTTP.

    Start the backend first:
        python rag_backend.py          # waits until fully ready, then serves on :8000

    Then start this MCP server (or let LM Studio manage it):
        python rag_server.py           # stdio  (default — for LM Studio)
        python rag_server.py --transport sse --port 8002  # SSE / remote

    The tool gracefully reports when the backend is offline rather than crashing.
"""

import sys
from pathlib import Path

import requests
from fastmcp import FastMCP

import log
import re
# ── Configuration ──────────────────────────────────────────────────────────────
BACKEND_URL   = "http://127.0.0.1:8000"
RETRIEVE_K    = 20
RETURN_K      = 5

# ── MCP Server ────────────────────────────────────────────────────────────────
mcp = FastMCP("LocalKnowledgeServer")

log.banner("LocalKnowledgeServer (MCP Proxy)", f"Backend: {BACKEND_URL}")

# Fetch available categories from the backend (best-effort, non-blocking)
_category_list = "(backend offline — start rag_backend.py first)"
try:
    resp = requests.get(f"{BACKEND_URL}/categories", timeout=3)
    if resp.ok:
        categories = resp.json().get("categories", [])
        _category_list = ", ".join(f"'{c}'" for c in categories) or "(none found)"
        log.info(f"Backend reachable. Available categories: {_category_list}")
    else:
        log.warn(f"Backend responded with {resp.status_code} — is it ready?")
except requests.exceptions.ConnectionError:
    log.warn("Backend not reachable yet. Start rag_backend.py before querying.")
except Exception as e:
    log.warn(f"Could not fetch categories from backend: {e}")


# ── MCP Tool ───────────────────────────────────────────────────────────────────

@mcp.tool()
def search_local_knowledge(query: str, category: str = None) -> str:
    f"""
Search the local cybersecurity knowledge base across ALL document categories.

Pipeline (runs in rag_backend.py):
  1. Retrieve top-{RETRIEVE_K} candidates via vector similarity search.
  2. Rerank candidates using a Cross-Encoder model.
  3. Return the top-{RETURN_K} highest-scoring results.

By default (when category is omitted), this searches the ENTIRE knowledge base
spanning all folders. Only pass a category if the user explicitly asks to filter.

Args:
    query:    The search string or question.
    category: Optional filter — restrict results to a single folder.
              Leave empty to search everything (preferred).
              Available categories: {_category_list}.

"""
    log.request("search_local_knowledge", {"query": query, "category": category})

    payload = {"query": query}
    if category:
        payload["category"] = category

    try:
        with log.Timer() as t:
            resp = requests.post(
                f"{BACKEND_URL}/search",
                json=payload,
                timeout=120,  # reranking can be slow on first cold call
            )

        if re.match(r"^5\\d{2}$", str(resp.status_code)):
            msg = "The knowledge base backend is still loading. Please try again in a moment."
            log.warn(msg)
            return msg

        if not resp.ok:
            err = resp.json().get("error", resp.text)
            log.error(f"Backend returned {resp.status_code}: {err}")
            return f"Backend error ({resp.status_code}): {err}"

        data    = resp.json()
        results = data.get("results", "No results returned.")
        backend_ms = data.get("elapsed_ms", 0)
        log.response("Backend", f"Search complete (backend: {backend_ms:.0f}ms, round-trip: {t.elapsed_ms:.0f}ms)")
        return results

    except requests.exceptions.ConnectionError:
        msg = (
            "Error: The RAG backend is not running. "
            "Please start it with: `python rag_backend.py`"
        )
        log.error(msg)
        return msg

    except requests.exceptions.Timeout:
        msg = "The search request timed out. The backend may be under load — please retry."
        log.error(msg)
        return msg

    except Exception as e:
        log.error(f"Unexpected error contacting backend: {e}")
        return f"An unexpected error occurred: {e}"


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LocalKnowledgeServer MCP Proxy")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport mode: 'stdio' (default) or 'sse' for network access",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (SSE mode)")
    parser.add_argument("--port", type=int, default=8002, help="Port (SSE mode, default: 8002)")
    args = parser.parse_args()

    if args.transport == "sse":
        log.info(f"MCP server starting on http://{args.host}:{args.port}/sse ...")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        log.info("MCP server starting on stdio...")
        mcp.run()
