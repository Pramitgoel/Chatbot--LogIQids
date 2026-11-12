from __future__ import annotations

from pathlib import Path

from gibberish_detector import detector

from ..query_types import QueryClassificationResult, QueryContext, QueryType
from .base import QueryDetector

_MODEL_PATH = Path(__file__).parent / "gibberish.model"
_GIBBERISH_DETECTOR: detector.Detector | None = None


def _get_gibberish_detector() -> detector.Detector | None:
    """Load model once per process; None if file missing."""
    global _GIBBERISH_DETECTOR
    if _GIBBERISH_DETECTOR is not None:
        return _GIBBERISH_DETECTOR
    if not _MODEL_PATH.is_file():
        return None

    _GIBBERISH_DETECTOR = detector.create_from_model(str(_MODEL_PATH))
    return _GIBBERISH_DETECTOR


class GibberishDetector(QueryDetector):
    """Gibberish via gibberish-detector; model loaded once per process."""

    def detect(self, context: QueryContext) -> QueryClassificationResult | None:
        text = (context.user_text or "").strip()
        if not text:
            return None
        det = _get_gibberish_detector()
        probability = 0.0
        if not det:
            return None
        probability = det.calculate_probability_of_being_gibberish(text)
        probability = round(probability, 2)

        if det.is_gibberish(text):
            return QueryClassificationResult(
                type=QueryType.GIBBERISH, explanation_text="", scores={"gibberish_probability": probability}
            )
        return QueryClassificationResult(
            type=QueryType.VALID_QUERY, explanation_text="", scores={"gibberish_probability": probability}
        )
