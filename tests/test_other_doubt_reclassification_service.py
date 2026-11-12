import json
import os
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "logiqids.settings")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("EMAIL_TRAKING_ENCRYPTION", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

import pytest

from content_package.v1.services.chatbot.chat_types import LLMResponse
from content_package.v1.services.chatbot.other_doubt_reclassification_evaluation import (
    evaluate_other_doubt_predictions,
)
from content_package.v1.services.chatbot.other_doubt_reclassification_service import (
    OtherDoubtCategory,
    OtherDoubtFallbackReason,
    OtherDoubtReclassificationService,
)
from content_package.v1.services.chatbot.query_types import QueryContext
from logiqids import settings as project_settings


class FakeProvider:
    def __init__(self, content, status_code=200, exc=None):
        self.content = content
        self.status_code = status_code
        self.exc = exc
        self.calls = []
        self.timeout = 30

    def generate_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return LLMResponse(
            content=self.content,
            raw={},
            status_code=self.status_code,
            usage={"total_tokens": 12, "prompt_tokens": 8, "completion_tokens": 4},
        )


def make_context(text, identifier="other"):
    return QueryContext(
        user_text=text,
        question_text="A sample reasoning question",
        solution_text="Official solution text",
        options=["10", "20", "40", "80"],
        identifier=identifier,
        question_id=1,
    )


@pytest.mark.parametrize(
    ("message", "category"),
    [
        ("Please explain the second step", OtherDoubtCategory.COULD_NOT_UNDERSTAND),
        ("The question does not provide enough information", OtherDoubtCategory.QUESTION_MAY_BE_WRONG),
        ("The answer should be 40, not 80", OtherDoubtCategory.ANSWER_MAY_BE_INCORRECT),
        ("There is a spelling mistake", OtherDoubtCategory.TYPING_MISTAKE),
        ("I don't understand why the answer is wrong", OtherDoubtCategory.COULD_NOT_UNDERSTAND),
        (
            "The answer is wrong because the question is missing a condition",
            OtherDoubtCategory.QUESTION_MAY_BE_WRONG,
        ),
    ],
)
def test_other_doubt_classifier_reclassifies_allowed_categories(message, category):
    provider = FakeProvider(json.dumps({"category": category.value, "confidence": 0.95, "reason_code": "MATCH"}))
    service = OtherDoubtReclassificationService(enabled=True, threshold=0.90, llm_provider=provider)

    result = service.classify(make_context(message))

    assert result.classified_category == category
    assert result.should_reclassify is True
    assert result.confidence == 0.95
    assert provider.calls[0]["temperature"] == 0
    assert provider.calls[0]["extra_params"] == {"response_format": {"type": "json_object"}}


@pytest.mark.parametrize("message", ["The app is not opening", "help"])
def test_other_doubt_classifier_retains_other(message):
    provider = FakeProvider(json.dumps({"category": "OTHER", "confidence": 0.93, "reason_code": "NOT_DOUBT"}))
    service = OtherDoubtReclassificationService(enabled=True, threshold=0.90, llm_provider=provider)

    result = service.classify(make_context(message))

    assert result.classified_category == OtherDoubtCategory.OTHER
    assert result.should_reclassify is False


def test_confidence_below_threshold_remains_other():
    provider = FakeProvider(
        json.dumps({"category": "ANSWER_MAY_BE_INCORRECT", "confidence": 0.89, "reason_code": "LOW_CONFIDENCE"})
    )
    service = OtherDoubtReclassificationService(enabled=True, threshold=0.90, llm_provider=provider)

    result = service.classify(make_context("The answer may be wrong"))

    assert result.classified_category == OtherDoubtCategory.OTHER
    assert result.should_reclassify is False
    assert result.fallback_reason == OtherDoubtFallbackReason.BELOW_THRESHOLD.value


def test_confidence_equal_threshold_reclassifies():
    provider = FakeProvider(
        json.dumps({"category": "ANSWER_MAY_BE_INCORRECT", "confidence": 0.90, "reason_code": "AT_THRESHOLD"})
    )
    service = OtherDoubtReclassificationService(enabled=True, threshold=0.90, llm_provider=provider)

    result = service.classify(make_context("The answer may be wrong"))

    assert result.classified_category == OtherDoubtCategory.ANSWER_MAY_BE_INCORRECT
    assert result.should_reclassify is True


def test_could_not_understand_question_reason_uses_question_breakdown_handler():
    provider = FakeProvider(
        json.dumps(
            {
                "category": "COULD_NOT_UNDERSTAND",
                "confidence": 0.95,
                "reason_code": "QUESTION_CLARIFICATION",
            }
        )
    )
    service = OtherDoubtReclassificationService(enabled=True, threshold=0.90, llm_provider=provider)

    result = service.classify(make_context("What does this question mean?"))

    assert result.handler_identifier == "could_not_understand_question"


def test_could_not_understand_solution_reason_uses_solution_handler():
    provider = FakeProvider(
        json.dumps(
            {
                "category": "COULD_NOT_UNDERSTAND",
                "confidence": 0.95,
                "reason_code": "SOLUTION_CLARIFICATION",
            }
        )
    )
    service = OtherDoubtReclassificationService(enabled=True, threshold=0.90, llm_provider=provider)

    result = service.classify(make_context("Please explain the second step"))

    assert result.handler_identifier == "could_not_understand_solution"


@pytest.mark.parametrize(
    ("content", "fallback_reason"),
    [
        (json.dumps({"category": "NOT_ALLOWED", "confidence": 0.99, "reason_code": "BAD"}), "INVALID_CATEGORY"),
        (json.dumps({"category": "OTHER", "confidence": "bad", "reason_code": "BAD"}), "INVALID_CONFIDENCE"),
        ("not json", "INVALID_JSON"),
    ],
)
def test_invalid_classifier_output_falls_back_to_other(content, fallback_reason):
    provider = FakeProvider(content)
    service = OtherDoubtReclassificationService(enabled=True, threshold=0.90, llm_provider=provider)

    result = service.classify(make_context("The answer should be 40"))

    assert result.classified_category == OtherDoubtCategory.OTHER
    assert result.should_reclassify is False
    assert result.fallback_reason == fallback_reason


@pytest.mark.parametrize("exc", [RuntimeError("api failed"), TimeoutError("timeout")])
def test_classifier_api_errors_fall_back_to_other(exc):
    provider = FakeProvider(None, exc=exc)
    service = OtherDoubtReclassificationService(enabled=True, threshold=0.90, llm_provider=provider)

    result = service.classify(make_context("The answer should be 40"))

    assert result.classified_category == OtherDoubtCategory.OTHER
    assert result.should_reclassify is False
    assert result.fallback_reason == OtherDoubtFallbackReason.API_ERROR.value


def test_classifier_does_not_run_for_non_other_original_category():
    provider = FakeProvider(json.dumps({"category": "ANSWER_MAY_BE_INCORRECT", "confidence": 0.99}))
    service = OtherDoubtReclassificationService(enabled=True, threshold=0.90, llm_provider=provider)

    result = service.classify(make_context("The answer should be 40", identifier="answer_incorrect"))

    assert result.classified_category == OtherDoubtCategory.OTHER
    assert result.should_reclassify is False
    assert result.fallback_reason == OtherDoubtFallbackReason.NOT_OTHER.value
    assert provider.calls == []


def test_feature_flag_disabled_keeps_other():
    provider = FakeProvider(json.dumps({"category": "ANSWER_MAY_BE_INCORRECT", "confidence": 0.99}))
    service = OtherDoubtReclassificationService(enabled=False, threshold=0.90, llm_provider=provider)

    result = service.classify(make_context("The answer should be 40"))

    assert result.classified_category == OtherDoubtCategory.OTHER
    assert result.should_reclassify is False
    assert result.fallback_reason == OtherDoubtFallbackReason.DISABLED.value
    assert provider.calls == []


def test_reclassification_numeric_settings_fall_back_safely(monkeypatch):
    monkeypatch.setenv("BAD_FLOAT_SETTING", "not-a-number")
    monkeypatch.setenv("BAD_RANGE_SETTING", "2")
    monkeypatch.setenv("GOOD_FLOAT_SETTING", "0.75")

    assert project_settings.get_float_env("BAD_FLOAT_SETTING", 0.9) == 0.9
    assert project_settings.get_float_env("BAD_RANGE_SETTING", 0.9, min_value=0, max_value=1) == 0.9
    assert project_settings.get_float_env("GOOD_FLOAT_SETTING", 0.9, min_value=0, max_value=1) == 0.75


def test_text_only_and_image_related_context_are_included_when_available():
    provider = FakeProvider(json.dumps({"category": "COULD_NOT_UNDERSTAND", "confidence": 0.95}))
    service = OtherDoubtReclassificationService(enabled=True, threshold=0.90, llm_provider=provider)

    service.classify(make_context("I cannot understand the diagram"))

    user_prompt = provider.calls[0]["messages"][1].content
    assert "Student message:" in user_prompt
    assert "Question text:" in user_prompt
    assert "Solution text:" in user_prompt
    assert "Options:" in user_prompt


def test_evaluation_fixture_reports_required_metrics():
    fixture_path = Path(__file__).parent / "fixtures" / "other_doubt_reclassification_eval.json"
    examples = json.loads(fixture_path.read_text())
    predictions = [example["expected"] for example in examples]

    report = evaluate_other_doubt_predictions(examples, predictions)

    assert report["total_examples"] == 20
    assert report["correct_classifications"] == 20
    assert report["incorrect_classifications"] == 0
    assert report["correctly_retained_as_other"] == 4
    assert report["false_reclassifications"] == 0
    assert report["results_by_category"]["OTHER"]["total"] == 4
    assert report["confusion_matrix"]["OTHER"]["OTHER"] == 4
