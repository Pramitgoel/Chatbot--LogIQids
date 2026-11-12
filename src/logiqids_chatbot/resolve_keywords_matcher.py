import re

from content_package.v1.constants.chatbot_constants import RESOLVE_KEYWORDS, RESOLVE_KEYWORDS_MATCH_THRESHOLD


class ResolveKeywordsMatcher:
    def __init__(self, keywords=None, threshold=None, only_exact_match=False):
        # Keywords are assumed already normalized (lowercase, no punctuation, single spaces)
        self.keywords = list(keywords) if keywords is not None else list(RESOLVE_KEYWORDS)
        self.threshold = threshold if threshold is not None else RESOLVE_KEYWORDS_MATCH_THRESHOLD
        self.only_exact_match = only_exact_match

    @staticmethod
    def normalize(text):
        if not text or not isinstance(text, str):
            return ""
        normalized = re.sub(r"[^\w\s]", " ", text.lower().strip())
        return " ".join(normalized.split())

    def should_resolve(self, user_text):
        if not user_text or not user_text.strip():
            return False
        normalized = self.normalize(user_text)
        if not normalized:
            return False

        # Exact match check
        if normalized in self.keywords:
            return True

        if self.only_exact_match:
            return False

        # Percentage match check: sum length of keywords that appear as substring / len(normalized)
        matched_len = sum(len(kw) for kw in self.keywords if kw in normalized)
        ratio = matched_len / len(normalized)
        return ratio >= self.threshold
