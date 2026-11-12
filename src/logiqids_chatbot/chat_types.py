from dataclasses import dataclass
from typing import Any, Literal

ChatRole = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    """One chat message; content is str or list of part dicts (e.g. image_url)."""

    role: ChatRole
    content: str | list[dict[str, Any]]


@dataclass
class LLMResponse:
    """LLM response: content text, raw payload, status_code, and token usage."""

    content: str | None
    raw: dict[str, Any]
    status_code: int
    usage: dict[str, Any] | None = None
