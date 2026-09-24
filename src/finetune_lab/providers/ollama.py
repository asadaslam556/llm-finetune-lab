"""Ollama provider, the default.

Talks to the local Ollama server's /api/chat with stream=False. Once the
export stage has deployed the fine-tune, the default model here becomes the
deployed name. Before that it is the stock base tag, which is how you get a
feel for the un-tuned behaviour to compare against.
"""

from __future__ import annotations

import time

import httpx

from ..core.config import get_settings
from .base import ChatResult, Provider, ProviderUnavailable


class OllamaProvider(Provider):
    name = "ollama"
    note = "Local models via Ollama. Free, private, needs `ollama serve` running."

    def __init__(self, host: str | None = None, timeout: float | None = None):
        s = get_settings()
        self.host = (host or s.ollama_host).rstrip("/")
        self.timeout = timeout or s.request_timeout_s

    def chat(self, messages: list[dict], model: str | None = None) -> ChatResult:
        s = get_settings()
        model = model or self.default_model()
        payload = {"model": model, "messages": messages, "stream": False}
        start = time.perf_counter()
        try:
            r = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except httpx.HTTPError as e:
            raise ProviderUnavailable(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` running? "
                f"({type(e).__name__})"
            ) from e
        if r.status_code == 404:
            # Ollama's way of saying it has never heard of that model tag.
            raise ProviderUnavailable(
                f"Ollama does not know the model '{model}'. Pull it first: "
                f"ollama pull {s.ollama_base_tag}"
            )
        if r.status_code >= 400:
            raise ProviderUnavailable(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")

        # Anything sitting between us and Ollama (a proxy, a captive portal, a
        # stray dev server on 11434) answers 200 with HTML. Treat that as an
        # outage so the chat panel shows a sentence rather than a 500.
        try:
            data = r.json()
        except ValueError as e:
            raise ProviderUnavailable(
                f"{self.host} replied with something that is not JSON. Is Ollama really on that port?"
            ) from e

        # Do not paper over a missing message with "". An empty chat bubble is
        # a silent failure, and the whole point of this layer is that failures
        # read like sentences.
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderUnavailable(
                f"Ollama returned no message content. Body starts: {str(data)[:160]}"
            )
        reply = message["content"]
        if not reply.strip():
            raise ProviderUnavailable(
                f"Ollama returned an empty reply for model '{model}'. That usually means the "
                "model loaded but produced nothing. Try a different prompt or re-pull the model."
            )
        return ChatResult(reply=reply, model=model, latency_ms=(time.perf_counter() - start) * 1000)

    def default_model(self) -> str:
        # Deployed fine-tune if the export stage has run, otherwise the base tag.
        s = get_settings()
        marker = s.artifacts_dir / "deployed.txt"
        if marker.exists():
            return marker.read_text(encoding="utf-8").strip()
        return s.ollama_base_tag
