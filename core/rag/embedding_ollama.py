"""
embedding_ollama.py

Ollama embedding client for RAG. Uses local Ollama (e.g. Jetson) at OLLAMA_BASE_URL
with model OLLAMA_EMBEDDING_MODEL (e.g. qwen3-embedding:8b). No API key or rate limit.
For slow devices (e.g. Jetson), use smaller api_batch_size and longer timeout.
"""

import logging
import os
import time
from typing import List, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_BASE_URL = "http://192.168.1.212:11434"
DEFAULT_MODEL = "qwen3-embedding:8b"


class OllamaEmbeddingClient:
    """
    Client for Ollama /api/embed. Exposes embed_texts, embed_query, embed_document.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        api_batch_size: int = 20,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        """
        :param base_url: Ollama server URL (default from OLLAMA_BASE_URL or 192.168.1.212:11434).
        :param model: Embedding model name (default qwen3-embedding:8b).
        :param api_batch_size: Texts per request; use 10-20 on Jetson to avoid timeout.
        :param timeout: Request timeout in seconds (default 300 for slow devices).
        :param max_retries: Retries on timeout or connection error.
        """
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.api_batch_size = max(1, api_batch_size)
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self._embed_url = f"{self.base_url}/api/embed"

    def embed_texts(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """
        Embed a list of texts via Ollama /api/embed. Batches into api_batch_size per request.
        Retries on timeout or connection error.
        """
        embeddings = []
        n = len(texts)
        for start in range(0, n, self.api_batch_size):
            batch = texts[start : start + self.api_batch_size]
            payload = {"model": self.model, "input": batch}
            last_err = None
            for attempt in range(self.max_retries + 1):
                try:
                    resp = requests.post(
                        self._embed_url,
                        json=payload,
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    embeddings.extend(data.get("embeddings", []))
                    logging.info("Embedded %s / %s chunks (Ollama local).", len(embeddings), n)
                    break
                except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, ConnectionError) as e:
                    last_err = e
                    if attempt < self.max_retries:
                        wait = 10 * (attempt + 1)
                        logging.warning("Ollama request timeout/error, retry in %ss (attempt %s/%s): %s", wait, attempt + 1, self.max_retries + 1, e)
                        time.sleep(wait)
                    else:
                        raise
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string (for retrieval)."""
        return self.embed_texts([query])[0]

    def embed_document(self, text: str) -> List[float]:
        """Embed a single document string."""
        return self.embed_texts([text])[0]
