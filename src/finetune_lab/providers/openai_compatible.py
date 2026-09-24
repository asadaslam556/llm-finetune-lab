"""Everything that speaks the OpenAI chat-completions shape.

OpenAI, DeepSeek, Together, Groq, OpenRouter, a local vLLM, and Ollama's own
/v1 endpoint all accept the same request body. So this is one class with a
different (key, base_url, model) triple per subclass, and adding a provider
is three lines rather than a new file.

DeepSeek is here rather than in its own module for exactly that reason: it is
the OpenAI API with a different host.
"""

from __future__ import annotations

import time

from ..core.config import get_settings
from .base import ChatResult, Provider, ProviderNotConfigured
from .http import parse, post


class OpenAICompatible(Provider):
    """Subclasses only need to say where their credentials come from."""

    # Human-facing name of the service, used in error messages.
    service = "OpenAI-compatible API"
    # Name of the Settings method returning (key, base_url, model).
    creds_attr = "openai"

    def _creds(self) -> tuple[str, str, str]:
        return getattr(get_settings(), self.creds_attr)()

    def service_name(self) -> str:
        return self.service

    def default_model(self) -> str:
        return self._creds()[2]

    def configured(self) -> bool:
        key, base_url, model = self._creds()
        return bool(key and base_url and model)

    def chat(self, messages: list[dict], model: str | None = None) -> ChatResult:
        s = get_settings()
        key, base_url, default_model = self._creds()
        who = self.service_name()
        if not key:
            raise ProviderNotConfigured(self.setup_hint())
        if not base_url:
            raise ProviderNotConfigured(f"{who} has no base URL set. Check your .env.")
        model = model or default_model

        start = time.perf_counter()
        r = post(
            f"{base_url}/chat/completions",
            json={"model": model, "messages": messages, "max_tokens": s.max_output_tokens},
            headers={"Authorization": f"Bearer {key}"},
            timeout=s.request_timeout_s,
            who=who,
        )
        reply = parse(r, lambda d: d["choices"][0]["message"]["content"], who)
        return ChatResult(reply=reply, model=model, latency_ms=(time.perf_counter() - start) * 1000)

    def setup_hint(self) -> str:
        var = f"{self.name.upper()}_API_KEY"
        return f"No key for {self.service_name()}. Set {var} (or LFL_{var}) in your .env."


class OpenAIProvider(OpenAICompatible):
    name = "openai"
    service = "the OpenAI API"
    creds_attr = "openai"
    note = "GPT models via the OpenAI API. Needs OPENAI_API_KEY."


class DeepSeekProvider(OpenAICompatible):
    name = "deepseek"
    service = "the DeepSeek API"
    creds_attr = "deepseek"
    note = "DeepSeek chat and reasoner models. Needs DEEPSEEK_API_KEY."


class CustomProvider(OpenAICompatible):
    """Any other OpenAI-compatible endpoint, configured purely from .env.

    This is the one that makes the repo useful to someone who clones it and
    uses neither of the two named services. Three env vars and they have a
    working provider without touching Python.
    """

    name = "custom"
    creds_attr = "custom"
    note = "Any OpenAI-compatible endpoint. Set CUSTOM_BASE_URL, CUSTOM_API_KEY, CUSTOM_MODEL."

    def service_name(self) -> str:
        return get_settings().custom_label

    def describe(self) -> str:
        _, base_url, model = self._creds()
        if not base_url:
            return self.note
        return f"{get_settings().custom_label} at {base_url} ({model or 'no model set'})."

    def setup_hint(self) -> str:
        return (
            "The custom provider needs CUSTOM_API_KEY, CUSTOM_BASE_URL and CUSTOM_MODEL. "
            "The base URL should end at /v1, the way OpenAI's does."
        )
