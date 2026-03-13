"""
rag_import_once.py

One-time RAG import: load documents from RAG_input (one folder per author; post .txt
and optional scripts/ folder with .py). Use --skip-embedding to only discover and
validate without calling the API or writing to the vector store.
By default runs incrementally: skips embedding for chunks already in the store.
Use --full to clear and reimport everything.
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from config.rag_config import get_rag_input_dir, get_chroma_dir, RAG_INPUT_DIR, RAG_CHROMA_DIR
from core.rag.document_loader import discover_posts, load_rag_documents
from core.rag.embedding_ollama import OllamaEmbeddingClient
from core.rag.vector_store import RAGVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="One-time RAG import from RAG_input to Chroma.")
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Only discover and list documents; do not call embedding API or write to Chroma",
    )
    parser.add_argument(
        "--rag-input",
        type=Path,
        default=None,
        help="Override RAG input directory (default: RAG_input from config)",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=None,
        help="Override Chroma persistence directory (default: from config / env RAG_CHROMA_DIR)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Approximate characters per chunk (default: 800)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlap between chunks (default: 100)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full reimport: clear collection and embed all chunks (default: incremental, skip already imported)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Chunks per batch written to Chroma (resume safe) (default: 50)",
    )
    parser.add_argument(
        "--max-chunks-per-run",
        type=int,
        default=0,
        help="Max chunks to embed this run (0 = no limit).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Ollama base URL (default: env OLLAMA_BASE_URL or http://192.168.1.212:11434)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="qwen3-embedding:8b",
        help="Ollama embedding model (default: qwen3-embedding:8b)",
    )
    args = parser.parse_args()

    rag_input = args.rag_input or get_rag_input_dir()
    if not rag_input.exists():
        logging.error("RAG input directory does not exist: %s", rag_input)
        sys.exit(1)

    posts = discover_posts(rag_input)
    if not posts:
        logging.warning("No documents found under %s (expected author folders with .txt or post/ and scripts/)", rag_input)
        sys.exit(0)

    authors = set(p["author"] for p in posts)
    n_posts = sum(1 for p in posts if p.get("doc_type") == "post")
    n_scripts = sum(1 for p in posts if p.get("doc_type") == "script")
    n_notebooks = sum(1 for p in posts if p.get("doc_type") == "notebook")
    logging.info(
        "Discovered %s documents from %s authors (posts: %s, scripts: %s, notebooks: %s)",
        len(posts),
        len(authors),
        n_posts,
        n_scripts,
        n_notebooks,
    )
    for a in sorted(authors):
        count = sum(1 for p in posts if p["author"] == a)
        logging.info("  Author: %s, documents: %s", a, count)

    chunks = list(load_rag_documents(rag_input, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap))
    logging.info("Total chunks for import: %s", len(chunks))

    if args.skip_embedding:
        chroma_dir = args.chroma_dir or get_chroma_dir()
        logging.info("Skip embedding and Chroma write (--skip-embedding). Chroma dir would be: %s", chroma_dir)
        return

    def make_chunk_id(c):
        """Stable id per chunk for incremental import (same file+chunk_id always gets same id)."""
        meta = c.get("metadata", {})
        key = f"{meta.get('source_file', '')}_{meta.get('chunk_id', 0)}"
        return hashlib.sha256(key.encode()).hexdigest()[:24]

    client = OllamaEmbeddingClient(
        base_url=args.ollama_url,
        model=args.ollama_model,
        api_batch_size=10,
        timeout=300,
    )
    logging.info("Using Ollama embedding: %s, model %s (batch %s, timeout 300s)", client.base_url, client.model, client.api_batch_size)

    store = RAGVectorStore(persist_directory=args.chroma_dir)

    if args.full:
        chunks_to_embed = chunks
        replace_first = True
        logging.info("Full reimport: embedding all %s chunks in batches of %s", len(chunks_to_embed), args.batch_size)
    else:
        existing_ids = store.get_existing_ids()
        chunks_to_embed = [c for c in chunks if make_chunk_id(c) not in existing_ids]
        skip_count = len(chunks) - len(chunks_to_embed)
        if skip_count:
            logging.info("Incremental: skipping %s already imported chunks, embedding %s new chunks", skip_count, len(chunks_to_embed))
        if not chunks_to_embed:
            logging.info("All chunks already in store. Nothing to embed. Use --full to reimport everything.")
            return
        replace_first = False

    if args.max_chunks_per_run > 0 and len(chunks_to_embed) > args.max_chunks_per_run:
        chunks_to_embed = chunks_to_embed[: args.max_chunks_per_run]
        logging.info("Limiting to %s chunks this run. Rerun without --full to continue.", len(chunks_to_embed))

    batch_size = min(max(1, args.batch_size), 10)
    logging.info("Ollama: using batch size %s per request.", batch_size)
    total = len(chunks_to_embed)
    for start in range(0, total, batch_size):
        batch = chunks_to_embed[start : start + batch_size]
        batch_ids = [make_chunk_id(c) for c in batch]
        batch_texts = [c["text"] for c in batch]
        replace = replace_first and (start == 0)
        logging.info("Embedding chunk %s to %s of %s (batch size %s)...", start + 1, start + len(batch), total, len(batch))
        embeddings = client.embed_texts(batch_texts)
        store.import_documents(batch, embeddings, ids=batch_ids, replace=replace)
    logging.info("Import complete. Vector store at: %s", store.persist_directory)


if __name__ == "__main__":
    main()
