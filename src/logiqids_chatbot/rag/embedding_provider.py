from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for embeddings; separate from LLM so RAG stays vendor-agnostic."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return embedding vector for text."""


class NotConfiguredEmbeddingProvider(EmbeddingProvider):
    """Placeholder when no embedding provider is configured."""

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("EmbeddingProvider is not configured.")
