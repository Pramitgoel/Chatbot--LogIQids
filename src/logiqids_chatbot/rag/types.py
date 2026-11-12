from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedDocument:
    """One retrieved doc: question_id, identifier, response_text, score, weightage (from response_bank)."""

    question_id: int
    identifier: str | None
    response_text: str
    score: float
    weightage: int


@dataclass
class RetrievalResult:
    """Retrieval result: documents (by score), confidence."""

    documents: list[RetrievedDocument]
    confidence: float
