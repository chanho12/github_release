"""OpenAI embedding adapter used by the Neo4j vector index."""
from __future__ import annotations

import os


class OpenAIEmbedder:
    def __init__(self, model: str | None = None, dimensions: int | None = None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.dimensions = dimensions or int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536"))
        self._client = None

    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def client(self):
        if not self.available():
            raise RuntimeError("OPENAI_API_KEY is required for vector GraphRAG")
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the openai package to create embeddings") from exc
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
