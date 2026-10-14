"""Pluggable LLM providers.

The default provider calls OpenAI's chat-completions API (needs
``OPENAI_API_KEY``). Tests and offline demos use ``StubLLMProvider``,
which is fully deterministic and makes no network calls.
"""
from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    """Anything that can turn a system+user prompt into a text answer."""

    name: str

    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...


class OpenAIProvider:
    """Chat-completions provider (``openai`` is imported lazily)."""

    name = "openai"

    def __init__(self, api_key: str | None, model: str = "gpt-4o-mini") -> None:
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it (or add it to a .env "
                "file) to enable answer generation."
            )
        self.api_key = api_key
        self.model = model

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


class StubLLMProvider:
    """Deterministic offline stand-in for tests and demos.

    It answers from the retrieved context embedded in the user prompt, so
    tests can exercise the full retrieval -> guardrails -> generation path
    without any network access.
    """

    name = "stub"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        context = user_prompt.split("Question:", 1)[0]
        snippet = " ".join(context.split())[:400]
        return (
            "Based on the indexed documents, here is what I found: "
            f"{snippet}..."
        )
