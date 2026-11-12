from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .chat_types import ChatMessage, LLMResponse


class LLMProvider(ABC):
    """Abstract LLM provider; small, vendor-agnostic interface."""

    @abstractmethod
    def generate_chat_completion(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate completion; extra_params for provider-specific options."""
