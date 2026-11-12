from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class QueryType(StrEnum):
    VALID_QUERY = "valid_query"
    IRRELEVANT = "irrelevant"
    GIBBERISH = "gibberish"
    ABUSE = "abuse"


@dataclass
class QueryContext:
    """Input for classification and RAG; provider-agnostic."""

    user_text: str
    question_text: str | None = None
    solution_text: str | None = None
    options: list[str] | None = None
    identifier: str | None = None
    history_snippet: str | None = None
    question_id: int | None = None
    query_embedding_vector: list[float] | None = None


@dataclass
class QueryClassificationResult:
    """Classification result: type, optional explanation_text, and detector scores for logging."""

    type: QueryType
    explanation_text: str | None = None
    scores: dict[str, float | None] = field(default_factory=dict)
