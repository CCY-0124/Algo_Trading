"""
vector_store.py

Chroma vector store for RAG. Supports one-time import: add all documents with
precomputed embeddings and persist to disk (e.g. external drive via RAG_CHROMA_DIR).
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.rag_config import get_chroma_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_chroma = None


def _get_chroma():
    global _chroma
    if _chroma is None:
        try:
            import chromadb
            _chroma = chromadb
        except ImportError:
            raise ImportError("Install chromadb: pip install chromadb")
    return _chroma


class RAGVectorStore:
    """
    Chroma-based vector store for one-time import and later retrieval.
    Persistence path is configurable (e.g. RAG_CHROMA_DIR on external drive).
    """

    COLLECTION_NAME = "rag_documents"

    def __init__(self, persist_directory: Optional[Path] = None):
        """
        :param persist_directory: Where to persist Chroma DB. Default from config (RAG_CHROMA_DIR).
        """
        self.persist_directory = Path(persist_directory) if persist_directory else get_chroma_dir()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None

    def _ensure_client(self):
        chroma = _get_chroma()
        if self._client is None:
            self._client = chroma.PersistentClient(path=str(self.persist_directory))
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "RAG documents from RAG_input (authors, posts, images)"},
            )
        return self._collection

    def get_existing_ids(self) -> set:
        """
        Return set of all document ids in the collection. Used for incremental import
        to skip chunks already embedded.
        """
        coll = self._ensure_client()
        n = coll.count()
        if n == 0:
            return set()
        result = coll.get(include=[], limit=n)
        ids = result.get("ids") or []
        return set(ids)

    def import_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        replace: bool = True,
    ) -> int:
        """
        One-time import: add documents with precomputed embeddings to Chroma.

        :param documents: List of dicts with 'text' and 'metadata'
        :param embeddings: List of embedding vectors (same length as documents)
        :param ids: Optional list of unique ids; if None, generated from index and metadata
        :param replace: If True, clear collection before adding (full reimport)
        :return: Number of documents added
        """
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings length must match")
        coll = self._ensure_client()
        if replace:
            try:
                self._client.delete_collection(name=self.COLLECTION_NAME)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "RAG documents from RAG_input"},
            )
            coll = self._collection
        if ids is None:
            ids = [f"doc_{i}_{doc.get('metadata', {}).get('post_basename', i)}" for i, doc in enumerate(documents)]
        texts = [d.get("text", "") for d in documents]
        metadatas = []
        for d in documents:
            meta = d.get("metadata", {})
            # Chroma accepts only str, int, float, bool in metadata
            clean = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            metadatas.append(clean)
        coll.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        logging.info("Imported %s documents into Chroma at %s", len(documents), self.persist_directory)
        return len(documents)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search by query embedding. Returns documents with metadata and distances.

        :param query_embedding: Query vector from GoogleEmbeddingClient.embed_query
        :param top_k: Number of results
        :param where: Optional Chroma metadata filter
        :return: List of dicts with document, metadata, distance
        """
        coll = self._ensure_client()
        result = coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for i in range(len(result.get("ids", [[]])[0])):
            out.append({
                "id": result["ids"][0][i],
                "document": result["documents"][0][i] if result.get("documents") else None,
                "metadata": (result["metadatas"][0][i] if result.get("metadatas") else None) or {},
                "distance": result["distances"][0][i] if result.get("distances") else None,
            })
        return out

    def count(self) -> int:
        """Return number of documents in the collection."""
        coll = self._ensure_client()
        return coll.count()
