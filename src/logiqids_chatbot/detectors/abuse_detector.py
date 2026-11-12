from __future__ import annotations

from collections.abc import Iterable

from better_profanity import profanity

from ..query_types import QueryClassificationResult, QueryContext, QueryType
from .base import QueryDetector


class ProfanityEngine:
    """Adapter for better_profanity."""

    def __init__(self, banned_words: Iterable[str] | None = []) -> None:
        words = [w for w in banned_words if w]
        if words:
            profanity.load_censor_words(words)
        else:
            profanity.load_censor_words()

    def is_abusive(self, text: str) -> bool:
        if not text:
            return False
        return profanity.contains_profanity(text)

    def get_abusive_words(self, text: str) -> list[str]:
        if not text:
            return []
        return profanity._regex.findall(text)


class AbuseDetector(QueryDetector):
    """Detects abuse/profanity via better_profanity."""

    def __init__(self, engine: ProfanityEngine) -> None:
        self.engine = engine

    def detect(self, context: QueryContext) -> QueryClassificationResult | None:
        if not context.user_text:
            return None
        matches = []
        if self.engine.is_abusive(context.user_text):
            matches = self.engine.get_abusive_words(context.user_text)
            return QueryClassificationResult(type=QueryType.ABUSE, explanation_text="", scores={"abuse_words": matches})
        return QueryClassificationResult(
            type=QueryType.VALID_QUERY, explanation_text="", scores={"abuse_words": matches}
        )
