from __future__ import annotations

from typing import Any

import requests

from content_package.v1.constants.chatbot_constants import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_IMAGE_MODEL,
    OPENAI_RELEVANCE_MODEL,
    OPENAI_TEXT_MODEL,
    OPENAI_TIMEOUT,
)
from content_package.v1.services.chatbot.chat_types import ChatMessage, LLMResponse
from content_package.v1.services.chatbot.llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI chat/completions; maps ChatMessage to payload, returns LLMResponse."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        text_model: str,
        image_model: str,
        relevance_model: str | None = None,
        timeout: float | None = None,
        extra_defaults: dict[str, Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.text_model = text_model
        self.image_model = image_model
        self.relevance_model = relevance_model or text_model
        self.timeout = timeout
        self.extra_defaults: dict[str, Any] = extra_defaults or {}

    @classmethod
    def from_settings(cls) -> OpenAIProvider:
        """Build provider from chatbot constants."""
        return cls(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            text_model=OPENAI_TEXT_MODEL,
            image_model=OPENAI_IMAGE_MODEL,
            relevance_model=OPENAI_RELEVANCE_MODEL,
            timeout=OPENAI_TIMEOUT,
            extra_defaults={},
        )

    def _build_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _select_default_model(self, messages: list[ChatMessage]) -> str:
        """Use image_model if any message has image content, else text_model."""
        for message in messages:
            if isinstance(message.content, list):
                for part in message.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return self.image_model
        return self.text_model

    def generate_chat_completion(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload_model = model or self._select_default_model(messages)
        body: dict[str, Any] = {
            "model": payload_model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
        }

        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        body.update(self.extra_defaults)
        if extra_params:
            body.update(extra_params)

        response = requests.post(
            self._build_url(),
            headers=self._build_headers(),
            json=body,
            timeout=self.timeout,
        )

        response_data = response.json()
        content: str | None = None
        usage: dict[str, Any] | None = None
        if response.status_code == 200:
            content = response_data["choices"][0]["message"]["content"]
            raw_usage = response_data.get("usage") or {}
            usage = {
                "total_tokens": raw_usage.get("total_tokens"),
                "prompt_tokens": raw_usage.get("prompt_tokens"),
                "completion_tokens": raw_usage.get("completion_tokens"),
                "cached_tokens": (raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            }

        return LLMResponse(
            content=content,
            raw=response_data,
            status_code=response.status_code,
            usage=usage,
        )
