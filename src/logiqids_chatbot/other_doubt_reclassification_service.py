from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from content_package.v1.constants.chatbot_constants import (
    CHATBOT_INSTRUCTION_BREAKDOWN_QUESTION,
    CHATBOT_INSTRUCTION_DEFEND_CORRECT_ANSWER,
    CHATBOT_INSTRUCTION_SIMPLIFY_SOLUTION,
    OTHER_DOUBT_RECLASSIFICATION_MODEL,
    OTHER_DOUBT_RECLASSIFICATION_SYSTEM_PROMPT,
    OTHER_DOUBT_RECLASSIFICATION_THRESHOLD,
    OTHER_DOUBT_RECLASSIFICATION_TIMEOUT,
)
from content_package.v1.services.chatbot.chat_types import ChatMessage
from content_package.v1.services.chatbot.factory import get_llm_provider
from content_package.v1.services.chatbot.llm_provider import LLMProvider
from content_package.v1.services.chatbot.query_types import QueryContext


class OtherDoubtCategory(StrEnum):
    COULD_NOT_UNDERSTAND = "COULD_NOT_UNDERSTAND"
    QUESTION_MAY_BE_WRONG = "QUESTION_MAY_BE_WRONG"
    ANSWER_MAY_BE_INCORRECT = "ANSWER_MAY_BE_INCORRECT"
    TYPING_MISTAKE = "TYPING_MISTAKE"
    OTHER = "OTHER"


class OtherDoubtFallbackReason(StrEnum):
    DISABLED = "DISABLED"
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    INVALID_CATEGORY = "INVALID_CATEGORY"
    INVALID_CONFIDENCE = "INVALID_CONFIDENCE"
    INVALID_JSON = "INVALID_JSON"
    API_ERROR = "API_ERROR"
    NOT_OTHER = "NOT_OTHER"


RECLASSIFIED_CATEGORY_TO_IDENTIFIER = {
    OtherDoubtCategory.QUESTION_MAY_BE_WRONG: "question_wrong",
    OtherDoubtCategory.ANSWER_MAY_BE_INCORRECT: "answer_incorrect",
    OtherDoubtCategory.TYPING_MISTAKE: "typo",
    OtherDoubtCategory.OTHER: "other",
}

COULD_NOT_UNDERSTAND_REASON_TO_IDENTIFIER = {
    "QUESTION_CLARIFICATION": "could_not_understand_question",
}

RECLASSIFIED_IDENTIFIER_TO_INSTRUCTIONS = {
    "could_not_understand_question": CHATBOT_INSTRUCTION_BREAKDOWN_QUESTION,
    "could_not_understand_solution": CHATBOT_INSTRUCTION_SIMPLIFY_SOLUTION,
    "answer_incorrect": CHATBOT_INSTRUCTION_DEFEND_CORRECT_ANSWER,
}

RECLASSIFIED_ADMIN_IDENTIFIERS = frozenset({"question_wrong", "typo"})


@dataclass
class OtherDoubtReclassificationResult:
    original_category: OtherDoubtCategory
    classified_category: OtherDoubtCategory
    confidence: float
    should_reclassify: bool
    reason_code: str
    handler_identifier: str
    fallback_reason: str | None = None
    latency_ms: int | None = None

    def to_log_data(self, final_handler_used: str | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "event": "other_doubt_reclassification",
            "original_category": self.original_category.value,
            "classified_category": self.classified_category.value,
            "confidence": self.confidence,
            "should_reclassify": self.should_reclassify,
            "reason_code": self.reason_code,
            "handler_identifier": self.handler_identifier,
            "final_handler_used": final_handler_used or self.handler_identifier,
        }
        if self.latency_ms is not None:
            data["classifier_latency_ms"] = self.latency_ms
        if self.fallback_reason:
            data["fallback_reason"] = self.fallback_reason
        return data


class OtherDoubtReclassificationService:
    """Conservative classifier for free-text doubts submitted under the visible Other option."""

    def __init__(
        self,
        enabled: bool = False,
        threshold: float = OTHER_DOUBT_RECLASSIFICATION_THRESHOLD,
        model: str = OTHER_DOUBT_RECLASSIFICATION_MODEL,
        timeout: float = OTHER_DOUBT_RECLASSIFICATION_TIMEOUT,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.enabled = enabled
        self.threshold = threshold
        self.model = model
        self.timeout = timeout
        self.llm_provider = llm_provider

    @classmethod
    def default(cls, enabled: bool = False) -> OtherDoubtReclassificationService:
        return cls(enabled=enabled)

    def classify(self, context: QueryContext) -> OtherDoubtReclassificationResult:
        """Classify a visible Other doubt, falling back to Other on any uncertainty."""
        if context.identifier != "other":
            return self._fallback_result(OtherDoubtFallbackReason.NOT_OTHER)
        if not self.enabled:
            return self._fallback_result(OtherDoubtFallbackReason.DISABLED)

        started_at = time.monotonic()
        classifier_payload, fallback_reason = self._request_classifier_payload(context)
        if fallback_reason:
            return self._fallback_result(fallback_reason, started_at=started_at)

        category = self._parse_category(classifier_payload.get("category"))
        if category is None:
            return self._fallback_result(OtherDoubtFallbackReason.INVALID_CATEGORY, started_at=started_at)

        confidence = self._parse_confidence(classifier_payload.get("confidence"))
        if confidence is None:
            return self._fallback_result(OtherDoubtFallbackReason.INVALID_CONFIDENCE, started_at=started_at)

        reason_code = self._parse_reason_code(classifier_payload.get("reason_code"))
        return self._result_from_valid_classifier_output(
            category, confidence, reason_code, self._latency_ms(started_at)
        )

    def _request_classifier_payload(
        self, context: QueryContext
    ) -> tuple[dict[str, Any], OtherDoubtFallbackReason | None]:
        try:
            response_text = self._request_classifier_output(context)
            return json.loads(response_text), None
        except json.JSONDecodeError:
            return {}, OtherDoubtFallbackReason.INVALID_JSON
        except Exception:
            return {}, OtherDoubtFallbackReason.API_ERROR

    def _result_from_valid_classifier_output(
        self,
        category: OtherDoubtCategory,
        confidence: float,
        reason_code: str,
        latency_ms: int,
    ) -> OtherDoubtReclassificationResult:
        if category == OtherDoubtCategory.OTHER:
            return self._classification_result(
                category, confidence, reason_code, reclassify=False, latency_ms=latency_ms
            )
        if confidence < self.threshold:
            return self._classification_result(
                OtherDoubtCategory.OTHER,
                confidence,
                reason_code,
                reclassify=False,
                fallback_reason=OtherDoubtFallbackReason.BELOW_THRESHOLD,
                latency_ms=latency_ms,
            )

        return self._classification_result(category, confidence, reason_code, reclassify=True, latency_ms=latency_ms)

    def _get_handler_identifier(self, category: OtherDoubtCategory, reason_code: str) -> str:
        if category == OtherDoubtCategory.COULD_NOT_UNDERSTAND:
            return COULD_NOT_UNDERSTAND_REASON_TO_IDENTIFIER.get(reason_code, "could_not_understand_solution")
        return RECLASSIFIED_CATEGORY_TO_IDENTIFIER[category]

    def _request_classifier_output(self, context: QueryContext) -> str:
        provider = self.llm_provider or get_llm_provider()
        original_timeout = getattr(provider, "timeout", None)
        if hasattr(provider, "timeout"):
            provider.timeout = self.timeout
        try:
            response = provider.generate_chat_completion(
                messages=[
                    ChatMessage(role="system", content=OTHER_DOUBT_RECLASSIFICATION_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=self._build_user_prompt(context)),
                ],
                model=self.model,
                temperature=0,
                max_tokens=80,
                extra_params={"response_format": {"type": "json_object"}},
            )
        finally:
            if hasattr(provider, "timeout"):
                provider.timeout = original_timeout
        if response.status_code != 200 or not response.content:
            raise RuntimeError("Classifier API returned no content")
        return response.content

    def _build_user_prompt(self, context: QueryContext) -> str:
        parts = [
            "Original selected category: OTHER",
            "Student message:",
            context.user_text or "",
        ]
        if context.question_text:
            parts.extend(["Question text:", context.question_text[:1000]])
        if context.solution_text:
            parts.extend(["Solution text:", context.solution_text[:1000]])
        if context.options:
            option_text = "\n".join([str(option)[:300] for option in context.options if option])
            if option_text:
                parts.extend(["Options:", option_text])
        return "\n\n".join(parts)

    def _fallback_result(
        self,
        fallback_reason: OtherDoubtFallbackReason,
        started_at: float | None = None,
    ) -> OtherDoubtReclassificationResult:
        return self._classification_result(
            OtherDoubtCategory.OTHER,
            0.0,
            fallback_reason.value,
            reclassify=False,
            fallback_reason=fallback_reason,
            latency_ms=self._latency_ms(started_at) if started_at else None,
        )

    def _classification_result(
        self,
        category: OtherDoubtCategory,
        confidence: float,
        reason_code: str,
        reclassify: bool,
        fallback_reason: OtherDoubtFallbackReason | None = None,
        latency_ms: int | None = None,
    ) -> OtherDoubtReclassificationResult:
        return OtherDoubtReclassificationResult(
            original_category=OtherDoubtCategory.OTHER,
            classified_category=category,
            confidence=confidence,
            should_reclassify=reclassify,
            reason_code=reason_code,
            handler_identifier=self._get_handler_identifier(category, reason_code),
            fallback_reason=fallback_reason.value if fallback_reason else None,
            latency_ms=latency_ms,
        )

    def _parse_category(self, value: Any) -> OtherDoubtCategory | None:
        try:
            return OtherDoubtCategory(str(value))
        except ValueError:
            return None

    def _parse_confidence(self, value: Any) -> float | None:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        if confidence < 0 or confidence > 1:
            return None
        return confidence

    def _parse_reason_code(self, value: Any) -> str:
        return str(value or "").strip()[:80] or "NO_REASON"

    def _latency_ms(self, started_at: float | None) -> int:
        if started_at is None:
            return 0
        return int((time.monotonic() - started_at) * 1000)
