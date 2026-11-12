from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..query_types import QueryClassificationResult, QueryContext, QueryType
from .base import QueryDetector

if TYPE_CHECKING:
    from ..rag.rag_service import RAGService


class RelevanceClient(Protocol):
    """RAG-based relevance: is query relevant to question/solution?"""

    def check_relevance(self, context: QueryContext) -> tuple[bool, float]: ...


class RagRelevanceClient:
    """Delegates to RAGService.check_relevance; returns (is_relevant, score)."""

    def __init__(self, rag_service: RAGService) -> None:
        self.rag_service = rag_service

    def check_relevance(self, context: QueryContext) -> tuple[bool, float]:
        is_relevant, score = self.rag_service.check_relevance(context)
        if not is_relevant:
            is_relevant_basic, score_basic = self.rag_service.relevance_strategy.is_relevant_with_basic_questions(
                context
            )
            if is_relevant_basic:
                return is_relevant_basic, score_basic
        return is_relevant, score


class RelevanceDetector(QueryDetector):
    """Uses RelevanceClient (e.g. RAG); missing/error => fail open (valid)."""

    def __init__(self, client: RagRelevanceClient | None = None) -> None:
        self.client = client

    def detect(self, context: QueryContext) -> QueryClassificationResult | None:
        """Return IRRELEVANT or VALID_QUERY with the relevance score; None only on error/no client."""
        if not self.client:
            return None
        is_relevant, score = self.client.check_relevance(context)
        score = round(score, 2)

        if not is_relevant:
            return QueryClassificationResult(
                type=QueryType.IRRELEVANT, explanation_text="", scores={"relevance_score": score}
            )
        return QueryClassificationResult(
            type=QueryType.VALID_QUERY, explanation_text="", scores={"relevance_score": score}
        )
