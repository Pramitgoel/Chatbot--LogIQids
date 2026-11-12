import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "logiqids.settings")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("EMAIL_TRAKING_ENCRYPTION", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

from content_package.v1.services.chatbot.other_doubt_reclassification_service import (
    OtherDoubtCategory,
    OtherDoubtReclassificationResult,
)
from content_package.v1.services.question_query_automation_service import QueryAutomation
from content_package.v4.services.question_query_chatbot_service import QuestionQueryChatBotService


class DummyQuery:
    id = 1
    query = "Other"
    identifier = "other"
    chatbot = True
    is_resolved = False


class DummyChat:
    id = 10
    chat_text = "Please explain the second step"


def make_result(category, should_reclassify=True, handler_identifier="could_not_understand_solution"):
    return OtherDoubtReclassificationResult(
        original_category=OtherDoubtCategory.OTHER,
        classified_category=category,
        confidence=0.95,
        should_reclassify=should_reclassify,
        reason_code="TEST",
        handler_identifier=handler_identifier,
    )


def make_service():
    service = QuestionQueryChatBotService.__new__(QuestionQueryChatBotService)
    service.question_id = 1
    service.user_id = 1
    service.profile_url = ""
    service.typing_enabled = False
    service.source = "question"
    service.query_obj = DummyQuery()
    return service


def test_reclassified_other_uses_existing_solution_prompt_arguments():
    service = make_service()
    result = make_result(OtherDoubtCategory.COULD_NOT_UNDERSTAND)
    service.get_chat_gpt_response = MagicMock(return_value="chat_obj")

    with patch("content_package.v4.services.question_query_chatbot_service.QueryBotLoggingService"):
        response = service._route_reclassified_other_doubt(result, DummyChat())

    assert response == "chat_obj"
    service.get_chat_gpt_response.assert_called_once_with(
        identifier="could_not_understand_solution",
        instructions="Simplify the solution for the student in brief.",
        routed_identifier="could_not_understand_solution",
    )


def test_reclassified_other_uses_existing_question_prompt_arguments():
    service = make_service()
    result = make_result(
        OtherDoubtCategory.COULD_NOT_UNDERSTAND,
        handler_identifier="could_not_understand_question",
    )
    service.get_chat_gpt_response = MagicMock(return_value="chat_obj")

    with patch("content_package.v4.services.question_query_chatbot_service.QueryBotLoggingService"):
        response = service._route_reclassified_other_doubt(result, DummyChat())

    assert response == "chat_obj"
    service.get_chat_gpt_response.assert_called_once_with(
        identifier="could_not_understand_question",
        instructions="Break down the question for the student in brief.",
        routed_identifier="could_not_understand_question",
    )


def test_reclassified_other_to_answer_incorrect_uses_defend_prompt():
    service = make_service()
    result = make_result(
        OtherDoubtCategory.ANSWER_MAY_BE_INCORRECT,
        handler_identifier="answer_incorrect",
    )
    service.get_chat_gpt_response = MagicMock(return_value="chat_obj")

    with patch("content_package.v4.services.question_query_chatbot_service.QueryBotLoggingService"):
        service._route_reclassified_other_doubt(result, DummyChat())

    service.get_chat_gpt_response.assert_called_once_with(
        identifier="answer_incorrect",
        instructions="Explain why the correct answer is correct.",
        routed_identifier="answer_incorrect",
    )


def test_reclassified_other_to_question_wrong_uses_existing_admin_handling():
    service = make_service()
    result = make_result(
        OtherDoubtCategory.QUESTION_MAY_BE_WRONG,
        handler_identifier="question_wrong",
    )
    service.send_query_to_admin = MagicMock(return_value="admin_chat_obj")

    with patch("content_package.v4.services.question_query_chatbot_service.QueryBotLoggingService"):
        response = service._route_reclassified_other_doubt(result, DummyChat())

    assert response == "admin_chat_obj"
    service.send_query_to_admin.assert_called_once()


def test_reclassified_other_to_typo_uses_existing_admin_handling():
    service = make_service()
    result = make_result(
        OtherDoubtCategory.TYPING_MISTAKE,
        handler_identifier="typo",
    )
    service.send_query_to_admin = MagicMock(return_value="admin_chat_obj")
    service.get_chat_gpt_response = MagicMock()

    with patch("content_package.v4.services.question_query_chatbot_service.QueryBotLoggingService"):
        response = service._route_reclassified_other_doubt(result, DummyChat())

    assert response == "admin_chat_obj"
    service.send_query_to_admin.assert_called_once()
    service.get_chat_gpt_response.assert_not_called()


def test_generic_other_prompt_remains_when_classification_does_not_reclassify():
    service = make_service()
    result = make_result(OtherDoubtCategory.OTHER, should_reclassify=False, handler_identifier="other")
    service.get_chat_gpt_response = MagicMock()

    assert result.should_reclassify is False
    service.get_chat_gpt_response.assert_not_called()


def test_original_and_internal_categories_are_logged_separately():
    result = make_result(
        OtherDoubtCategory.ANSWER_MAY_BE_INCORRECT,
        handler_identifier="answer_incorrect",
    )

    log_data = result.to_log_data(final_handler_used="answer_incorrect")

    assert log_data["original_category"] == "OTHER"
    assert log_data["classified_category"] == "ANSWER_MAY_BE_INCORRECT"
    assert log_data["handler_identifier"] == "answer_incorrect"
    assert log_data["final_handler_used"] == "answer_incorrect"


def test_automation_receives_routed_identifier_for_existing_specialized_mode():
    service = make_service()
    service.get_response_from_past_responses = MagicMock(return_value="")
    service.log_gpt_response = MagicMock()
    service.add_chat = MagicMock(return_value="chat_obj")
    queryset = MagicMock()
    queryset.count.return_value = 0
    queryset.__iter__.return_value = iter([])

    with patch("content_package.v4.services.question_query_chatbot_service.QuestionQueriesChats") as chats:
        chats.objects.filter.return_value = queryset
        with patch("content_package.v4.services.question_query_chatbot_service.QueryAutomation") as automation:
            automation.return_value.generate_response.return_value = "GPT response"
            service.get_chat_gpt_response(
                identifier="answer_incorrect",
                instructions="Explain why the correct answer is correct.",
                routed_identifier="answer_incorrect",
            )

    assert automation.call_args.kwargs["routed_identifier"] == "answer_incorrect"


def test_reclassified_routed_identifiers_select_existing_query_automation_modes():
    query_obj = MagicMock()
    query_obj.identifier = "other"

    automation = QueryAutomation(query_id=1, routed_identifier="could_not_understand_question")
    automation.set_current_mode(query_obj)
    assert automation.current_mode == "QUESTION"

    automation = QueryAutomation(query_id=1, routed_identifier="could_not_understand_solution")
    automation.set_current_mode(query_obj)
    assert automation.current_mode == "SOLUTION"

    automation = QueryAutomation(query_id=1, routed_identifier="answer_incorrect")
    automation.set_current_mode(query_obj)
    assert automation.current_mode == "DEFEND_CORRECT_ANSWER"


def test_existing_specialized_prompt_selection_remains_unchanged_for_direct_categories():
    query_obj = MagicMock()

    query_obj.identifier = "could_not_understand_question"
    automation = QueryAutomation(query_id=1)
    automation.set_current_mode(query_obj)
    assert automation.current_mode == "QUESTION"

    query_obj.identifier = "could_not_understand_solution"
    automation = QueryAutomation(query_id=1)
    automation.set_current_mode(query_obj)
    assert automation.current_mode == "SOLUTION"

    query_obj.identifier = "answer_incorrect"
    automation = QueryAutomation(query_id=1)
    automation.set_current_mode(query_obj)
    assert automation.current_mode == "DEFEND_CORRECT_ANSWER"
