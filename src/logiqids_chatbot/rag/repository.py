from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from content_package.models import QuestionEmbedding
from content_package.v1.constants.chatbot_constants import RAG_WEIGHTAGE_ALPHA

from .types import RetrievedDocument


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors; 0 if invalid."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingRepository(ABC):
    """Vector store for question embeddings."""

    @abstractmethod
    def store(
        self,
        question_id: int,
        response_bank_id: int | None,
        embedding_vector: Sequence[float],
    ) -> QuestionEmbedding:
        """Store one embedding; returns created QuestionEmbedding."""

    @abstractmethod
    def search(
        self,
        question_id: int,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievedDocument]:
        """Search similar embeddings; returns top_k by score."""


class DjangoEmbeddingRepository(EmbeddingRepository):
    """QuestionEmbedding ORM; score = cosine_sim + alpha * normalized_weightage."""

    def __init__(self, weightage_alpha: float | None = None) -> None:
        self.weightage_alpha = weightage_alpha if weightage_alpha is not None else RAG_WEIGHTAGE_ALPHA

    def store(
        self,
        question_id: int,
        response_bank_id: int | None,
        embedding_vector: Sequence[float],
    ) -> QuestionEmbedding:
        return QuestionEmbedding.objects.create(
            question_id=question_id,
            embedding_vector=list(embedding_vector),
            response_bank_id=response_bank_id,
        )

    def search(
        self,
        question_id: int,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievedDocument]:
        qs = QuestionEmbedding.objects.filter(
            question_id=question_id, response_bank__isnull=False, is_active=True
        ).select_related("response_bank")
        embeddings = list(qs)
        if not embeddings:
            return []

        weights = [
            abs(e.response_bank.weightage)
            for e in embeddings
            if e.response_bank and e.response_bank.weightage is not None
        ]
        max_weight = max(weights) if weights else 1

        scored: list[RetrievedDocument] = []
        for e in embeddings:
            sim = _cosine_similarity(query_vector, e.embedding_vector or [])
            rb = e.response_bank
            weight_val = rb.weightage if rb and rb.weightage is not None else 0
            weight_norm = weight_val / max_weight if max_weight else 0.0
            final_score = sim + self.weightage_alpha * weight_norm
            identifier = rb.identifier if rb else None
            response_text = rb.response if rb else ""
            scored.append(
                RetrievedDocument(
                    question_id=e.question_id,
                    identifier=identifier,
                    response_text=response_text,
                    score=final_score,
                    weightage=weight_val,
                )
            )

        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]
