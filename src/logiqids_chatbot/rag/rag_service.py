from __future__ import annotations

from content_package.v1.constants.chatbot_constants import (
    OPENAI_API_KEY,
    RAG_CONFIDENCE_THRESHOLD,
    RAG_TOP_K,
)

from ..query_types import QueryContext
from .embedding_provider import EmbeddingProvider, NotConfiguredEmbeddingProvider
from .openai_embedding_provider import OpenAIEmbeddingProvider
from .repository import DjangoEmbeddingRepository, EmbeddingRepository
from .strategies.relevance import RagRelevanceStrategy, RelevanceStrategy
from .strategies.retrieval import QuestionRagRetrievalStrategy, RetrievalStrategy
from .types import RetrievalResult


class RAGService:
    """RAG: relevance check and hallucination-safe retrieval."""

    def __init__(
        self,
        repository: EmbeddingRepository,
        embedding_provider: EmbeddingProvider,
        relevance_strategy: RelevanceStrategy,
        retrieval_strategy: RetrievalStrategy,
        confidence_threshold: float,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.relevance_strategy = relevance_strategy
        self.retrieval_strategy = retrieval_strategy
        self.confidence_threshold = confidence_threshold

    @classmethod
    def default(cls) -> RAGService:
        """Default RAGService with dependencies from chatbot constants."""
        repository = DjangoEmbeddingRepository()
        embedding_provider: EmbeddingProvider = (
            OpenAIEmbeddingProvider() if OPENAI_API_KEY else NotConfiguredEmbeddingProvider()
        )
        top_k = RAG_TOP_K
        confidence_threshold = RAG_CONFIDENCE_THRESHOLD

        relevance_strategy = RagRelevanceStrategy(
            repository=repository,
            embedding_provider=embedding_provider,
        )
        retrieval_strategy = QuestionRagRetrievalStrategy(
            repository=repository,
            embedding_provider=embedding_provider,
            top_k=top_k,
        )
        return cls(
            repository=repository,
            embedding_provider=embedding_provider,
            relevance_strategy=relevance_strategy,
            retrieval_strategy=retrieval_strategy,
            confidence_threshold=confidence_threshold,
        )

    def check_relevance(self, context: QueryContext) -> tuple[bool, float]:
        """Return (is_relevant, score) via relevance strategy."""
        return self.relevance_strategy.is_relevant(context)

    def get_answer_with_rag(
        self,
        context: QueryContext,
    ) -> tuple[str | None, bool, RetrievalResult]:
        """Retrieve; if confidence >= threshold return top response, else (None, False, result)."""
        retrieval_result = self.retrieval_strategy.retrieve(context)
        if retrieval_result.confidence >= self.confidence_threshold and retrieval_result.documents:
            return retrieval_result.documents[0].response_text, True, retrieval_result

        return None, False, retrieval_result
