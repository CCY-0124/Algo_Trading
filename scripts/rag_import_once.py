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
from pathlib import Path

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so GEMINI_API_KEY or GOOGLE_EMBEDDING_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from config.rag_config import get_rag_input_dir, get_chroma_dir, RAG_INPUT_DIR, RAG_CHROMA_DIR
from core.rag.document_loader import discover_posts, load_rag_documents
from core.rag.embedding_google import GoogleEmbeddingClient
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
        "--api-batch-size",
        type=int,
        default=100,
        help="Chunks per API request (limit is per request; 100 = ~140 requests for 14k chunks) (default: 100)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between batch API requests (default: 0.5)",
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
        help="Max chunks to embed this run (0 = no limit). Not needed when using batch API.",
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

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_EMBEDDING_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY or GOOGLE_EMBEDDING_API_KEY is not set. Set one in .env to run import.")
        sys.exit(1)

    client = GoogleEmbeddingClient(api_batch_size=args.api_batch_size, delay_between_batches=args.delay)
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
        logging.info(
            "Limiting to %s chunks this run (free tier ~1000/day). Rerun tomorrow without --full to continue.",
            len(chunks_to_embed),
        )

    batch_size = max(1, args.batch_size)
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
