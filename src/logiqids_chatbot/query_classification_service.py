from __future__ import annotations

from .detectors.abuse_detector import AbuseDetector, ProfanityEngine
from .detectors.base import QueryDetector
from .detectors.gibberish_detector import GibberishDetector
from .detectors.relevance_detector import RagRelevanceClient, RelevanceDetector
from .query_types import QueryClassificationResult, QueryContext, QueryType
from .rag.rag_service import RAGService


class QueryClassificationService:
    """Chain of detectors (abuse → gibberish → relevance); first match wins, else valid."""

    def __init__(self, detectors: list[QueryDetector]) -> None:
        self.detectors = detectors

    def classify(self, context: QueryContext) -> QueryClassificationResult:
        scores = {}
        for detector in self.detectors:
            if not detector.supports(context):
                continue
            result = detector.detect(context)
            if result is not None:
                if isinstance(result, QueryClassificationResult):
                    if result.type != QueryType.VALID_QUERY:
                        return result
                    scores.update(result.scores)
        return QueryClassificationResult(type=QueryType.VALID_QUERY, explanation_text=None, scores=scores)

    @classmethod
    def default(cls) -> QueryClassificationService:
        """Default chain: AbuseDetector, GibberishDetector, RelevanceDetector (RAG)."""
        profanity_engine = ProfanityEngine()
        rag_service = RAGService.default()
        relevance_client = RagRelevanceClient(rag_service)

        detectors: list[QueryDetector] = [
            AbuseDetector(engine=profanity_engine),
            GibberishDetector(),
            RelevanceDetector(client=relevance_client),
        ]
        return cls(detectors=detectors)
