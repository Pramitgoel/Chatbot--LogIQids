from __future__ import annotations

from abc import ABC, abstractmethod

from ..query_types import QueryClassificationResult, QueryContext


class QueryDetector(ABC):
    """Strategy for one classification concern (abuse, gibberish, relevance)."""

    def supports(self, context: QueryContext) -> bool:
        """Whether to run for this context; default True."""
        return True

    @abstractmethod
    def detect(self, context: QueryContext) -> QueryClassificationResult | None:
        """Return result to short-circuit chain, or None to pass to next detector."""
