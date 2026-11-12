from __future__ import annotations

from content_package.v1.constants.chatbot_constants import LLM_PROVIDER

from .llm_provider import LLMProvider
from .providers.openai_provider import OpenAIProvider


def get_llm_provider(provider_name: str | None = None) -> LLMProvider:
    """Resolve active LLM provider from name or constants; supports 'openai'."""
    name = (provider_name or LLM_PROVIDER or "openai").strip().lower()
    if name == "openai":
        return OpenAIProvider.from_settings()

    raise NotImplementedError(f"Unsupported LLM provider: {name}")
