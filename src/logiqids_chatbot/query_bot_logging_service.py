from typing import Any

from content_package.models import QueryBotLogs, QuestionQueriesChats
from content_package.v1.services.chatbot.chat_types import LLMResponse
from content_package.v1.services.chatbot.query_types import QueryClassificationResult
from content_package.v1.services.chatbot.rag.types import RetrievalResult


class QueryBotLoggingService:
    @staticmethod
    def log_classification(chat_obj: QuestionQueriesChats, classification_result: QueryClassificationResult) -> None:
        data: dict[str, Any] = {}
        if classification_result.scores:
            data.update({k: v for k, v in classification_result.scores.items() if v is not None})
        QueryBotLoggingService.log_query_bot_response(chat_obj, classification_result.type.value, data)

    @staticmethod
    def log_response_bank_solution(chat_obj: QuestionQueriesChats, match_score: float | None) -> None:
        data: dict[str, Any] = {"response_bank_match_score": match_score}
        QueryBotLoggingService.log_query_bot_response(chat_obj, "response_bank", data)

    @staticmethod
    def log_rag_solution(chat_obj: QuestionQueriesChats, retrieval_result: RetrievalResult) -> None:
        retrieval_scores: list[float] = [doc.score for doc in retrieval_result.documents]
        data: dict[str, Any] = {
            "rag_match_score": retrieval_result.confidence,
            "rag_retrieval_scores": retrieval_scores,
        }
        QueryBotLoggingService.log_query_bot_response(chat_obj, "rag", data)

    @staticmethod
    def log_llm_solution(
        chat_obj: QuestionQueriesChats,
        llm_response: LLMResponse,
        rag_match_score: float | None = None,
        rag_retrieval_scores: list[float] | None = None,
    ) -> None:
        usage = llm_response.usage or {}
        data: dict[str, Any] = {
            "llm_total_tokens": usage.get("total_tokens"),
            "llm_prompt_tokens": usage.get("prompt_tokens"),
            "llm_completion_tokens": usage.get("completion_tokens"),
            "llm_cached_tokens": usage.get("cached_tokens"),
        }
        if rag_match_score is not None:
            data["rag_match_score"] = rag_match_score
        if rag_retrieval_scores is not None:
            data["rag_retrieval_scores"] = rag_retrieval_scores

        QueryBotLoggingService.log_query_bot_response(chat_obj, "llm", data)

    @staticmethod
    def log_admin_response_solution(chat_obj: QuestionQueriesChats, admin_id: int | None = None) -> None:
        data: dict[str, Any] = {}
        if admin_id is not None:
            data["admin_id"] = admin_id

        QueryBotLoggingService.log_query_bot_response(chat_obj, "admin_response", data)

    @staticmethod
    def log_other_doubt_reclassification(chat_obj: QuestionQueriesChats, data: dict[str, Any]) -> None:
        QueryBotLoggingService.log_query_bot_response(chat_obj, "valid_query", data)

    @staticmethod
    def log_query_bot_response(chat_obj: QuestionQueriesChats, response_type: str, data: dict[str, Any]) -> None:
        QueryBotLogs.objects.create(chat=chat_obj, response_type=response_type, data=data)
