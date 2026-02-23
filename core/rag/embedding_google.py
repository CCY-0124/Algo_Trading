"""
embedding_google.py

Google Embedding API client for RAG. The daily limit is per HTTP request, not per chunk.
embed_content accepts a list of strings; we send batches (e.g. 100 chunks per request)
so 13,000 chunks use ~130 requests, well under the free tier.
"""

import logging
import os
import re
import time
from typing import List, Optional

from config.rag_config import GEMINI_API_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_client = None


def _get_client(api_key: Optional[str] = None):
    global _client
    if _client is None:
        try:
            from google import genai
            key = api_key or GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_EMBEDDING_API_KEY")
            _client = genai.Client(api_key=key) if key else genai.Client()
        except ImportError:
            raise ImportError("Install the Google GenAI SDK: pip install google-genai")
    return _client


class GoogleEmbeddingClient:
    """
    Client for Google embedding API (text-embedding-004). Uses GEMINI_API_KEY
    or GOOGLE_EMBEDDING_API_KEY from environment.
    """

    # Gemini API embedding model (use gemini-embedding-001 if text-embedding-* 404)
    DEFAULT_MODEL = "gemini-embedding-001"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        api_batch_size: int = 100,
        delay_between_batches: float = 0.5,
    ):
        """
        :param api_key: Google API key. If None, uses GEMINI_API_KEY or GOOGLE_EMBEDDING_API_KEY from env.
        :param model: Embedding model name (default gemini-embedding-001).
        :param api_batch_size: Texts per API request (daily limit is per request; 100 chunks/request = 130 requests for 13k chunks).
        :param delay_between_batches: Seconds between batch requests to avoid rate limits.
        """
        self._api_key = api_key or GEMINI_API_KEY
        self.model = model
        self.api_batch_size = max(1, min(api_batch_size, 250))
        self.delay_between_batches = delay_between_batches
        if not self._api_key:
            logging.warning("GEMINI_API_KEY / GOOGLE_EMBEDDING_API_KEY not set; embedding calls may fail.")

    def embed_texts(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """
        Embed a list of texts. Sends multiple texts per API request (batch); daily limit
        is per request, so e.g. 100 chunks per request means 130 requests for 13,000 chunks.

        :param texts: List of strings to embed.
        :param task_type: RETRIEVAL_DOCUMENT for documents, RETRIEVAL_QUERY for queries.
        :return: List of embedding vectors (same order as texts).
        """
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_EMBEDDING_API_KEY is not set. Set it in .env or environment.")
        from google.genai import types
        from google.genai.errors import ClientError
        client = _get_client(self._api_key)
        config = types.EmbedContentConfig(task_type=task_type)
        embeddings = []
        n = len(texts)
        max_attempts = 5
        for start in range(0, n, self.api_batch_size):
            if start > 0:
                time.sleep(self.delay_between_batches)
            batch = texts[start : start + self.api_batch_size]
            for attempt in range(max_attempts):
                try:
                    result = client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=config,
                    )
                    for emb in result.embeddings:
                        embeddings.append(list(emb.values))
                    logging.info("Embedded %s / %s chunks (%s requests so far).", len(embeddings), n, (start // self.api_batch_size) + 1)
                    break
                except ClientError as e:
                    err_str = str(e)
                    is_rate_limit = getattr(e, "status_code", None) == 429 or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                    if is_rate_limit and attempt < max_attempts - 1:
                        match = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", err_str, re.I)
                        wait = int(float(match.group(1))) + 1 if match else 45
                        logging.warning("Rate limit (429), waiting %ss before retry (attempt %s/%s)...", wait, attempt + 1, max_attempts)
                        time.sleep(wait)
                    else:
                        raise
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string (for retrieval)."""
        return self.embed_texts([query], task_type="RETRIEVAL_QUERY")[0]

    def embed_document(self, text: str) -> List[float]:
        """Embed a single document string."""
        return self.embed_texts([text], task_type="RETRIEVAL_DOCUMENT")[0]
