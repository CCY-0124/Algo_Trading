"""
rag_ui.py

Streamlit UI for RAG knowledge base search.
Uses Ollama (local) embedding backend.

Run:
    streamlit run scripts/rag_ui.py
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import streamlit as st

from config.rag_config import get_chroma_dir, get_rag_input_dir
from core.rag.document_loader import load_full_document_by_source
from core.rag.vector_store import RAGVectorStore

st.set_page_config(
    page_title="RAG Knowledge Search",
    page_icon="",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------------

st.sidebar.title("Settings")

ollama_url = st.sidebar.text_input(
    "Ollama URL",
    value=os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.212:11434"),
)
ollama_model = st.sidebar.text_input(
    "Embedding model",
    value="qwen3-embedding:4b",
    help="Must match import model. 4b=2560 dims, 8b=4096 dims.",
)

top_k = st.sidebar.slider("Number of results (top-k)", min_value=1, max_value=20, value=5)

author_filter = st.sidebar.selectbox(
    "Filter by author (optional)",
    ["All", "Python 股票量化分析週記", "蔡嘉民 Calvin", "錢琛 Chin Shum"],
)

doc_type_filter = st.sidebar.selectbox(
    "Filter by document type (optional)",
    ["All", "post", "script", "notebook"],
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Connecting to vector store...")
def load_store():
    return RAGVectorStore(persist_directory=get_chroma_dir())


@st.cache_resource(show_spinner="Loading embedding client...")
def load_embedding_client(url: str, model: str):
    from core.rag.embedding_ollama import OllamaEmbeddingClient
    return OllamaEmbeddingClient(base_url=url, model=model, api_batch_size=1, timeout=120)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("RAG Knowledge Base Search")
st.caption(f"Vector store: {get_chroma_dir()}")

store = load_store()
total_docs = store.count()
expected_dim = store.get_embedding_dimension()
st.sidebar.metric("Total chunks in DB", total_docs)
if expected_dim:
    st.sidebar.caption(f"Expected embedding dim: {expected_dim} (use matching model)")

if total_docs == 0:
    st.warning(
        "Vector store is empty. Run `python scripts/rag_import_once.py --embedding ollama --full` first."
    )
    st.stop()

query = st.text_input("Search query", placeholder="e.g. RSI momentum strategy in crypto market")

search_clicked = st.button("Search", type="primary", disabled=not query.strip())

if search_clicked and query.strip():
    client = load_embedding_client(ollama_url, ollama_model)

    with st.spinner("Embedding query and searching..."):
        try:
            query_vec = client.embed_query(query.strip())
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            st.stop()

        if expected_dim and len(query_vec) != expected_dim:
            st.error(
                f"Embedding dimension mismatch: collection expects {expected_dim}, "
                f"got {len(query_vec)}. Use the same model as import "
                f"(2560=qwen3-embedding:4b, 4096=qwen3-embedding:8b)."
            )
            st.stop()

        where_filter = {}
        if author_filter != "All":
            where_filter["author"] = author_filter
        if doc_type_filter != "All":
            where_filter["doc_type"] = doc_type_filter

        try:
            results = store.search(
                query_embedding=query_vec,
                top_k=top_k,
                where=where_filter if where_filter else None,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "dimension" in err_str or "invalidargument" in type(e).__name__.lower():
                st.error(
                    f"Embedding dimension mismatch: {e} "
                    "Use the same model as import (4b=2560 dims, 8b=4096 dims)."
                )
            else:
                st.error(f"Search failed: {e}")
            st.stop()

    if not results:
        st.info("No results found. Try a different query or remove filters.")
    else:
        seen_source = set()
        unique_results = []
        for r in results:
            sf = r.get("metadata", {}).get("source_file", "")
            if sf and sf in seen_source:
                continue
            if sf:
                seen_source.add(sf)
            unique_results.append(r)

        st.success(f"Found {len(unique_results)} results")
        for i, r in enumerate(unique_results):
            meta = r.get("metadata", {})
            author = meta.get("author", "Unknown")
            doc_type = meta.get("doc_type", "")
            file_name = meta.get("file_name", "")
            distance = r.get("distance")
            score_label = f"{1 - distance:.3f}" if distance is not None else "N/A"

            with st.expander(
                f"#{i + 1}  [{author}] {file_name}  ({doc_type})  similarity: {score_label}",
                expanded=(i == 0),
            ):
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"**Author:** {author}")
                col2.markdown(f"**Type:** {doc_type}")
                col3.markdown(f"**File:** {file_name}")

                st.divider()
                matched_chunk = r.get("document", "")
                full_doc = load_full_document_by_source(meta.get("source_file", ""), get_rag_input_dir())

                if full_doc and full_doc != matched_chunk:
                    with st.expander("Matched chunk (what triggered this result)", expanded=False):
                        st.markdown(matched_chunk)
                    st.markdown("**Full document:**")
                    st.markdown(full_doc)
                else:
                    st.markdown(matched_chunk)
