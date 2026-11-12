import requests

from content_package.v1.constants.chatbot_constants import OPENAI_API_KEY

from .embedding_provider import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings API."""

    def __init__(self, api_key=None, model="text-embedding-3-small"):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.url = "https://api.openai.com/v1/embeddings"

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            return []
        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"input": text[:8191], "model": self.model},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
