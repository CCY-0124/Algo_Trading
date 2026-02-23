"""
core.rag

RAG (Retrieval Augmented Generation) components:
- Document loading from RAG_input (one folder per author; post .txt and sample .py in scripts/)
- Google embedding API client
- Chroma vector store with one-time import support
"""

from .document_loader import load_rag_documents, discover_posts
from .embedding_google import GoogleEmbeddingClient
from .vector_store import RAGVectorStore

__all__ = [
    "load_rag_documents",
    "discover_posts",
    "GoogleEmbeddingClient",
    "RAGVectorStore",
]
