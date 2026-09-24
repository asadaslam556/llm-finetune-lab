"""Anthropic provider (Claude).

Points at api.anthropic.com by default. Set LFL_ANTHROPIC_BASE_URL (or the
standard ANTHROPIC_BASE_URL) to route through a gateway instead, which is how
most corporate deployments work. The request shape is identical either way,
so nothing else in the app changes.

The one thing that makes Anthropic different from the OpenAI-shaped APIs is
that the system prompt is a top-level field rather than a message with
role="system".
"""

from __future__ import annotations

import time

from ..core.config import get_settings
from .base import ChatResult, Provider, ProviderNotConfigured
from .http import parse, post


def _text_blocks(data: dict) -> str:
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


class AnthropicProvider(Provider):
    name = "anthropic"
    note = "Claude via the Anthropic Messages API. Works with a gateway base URL too."

    def default_model(self) -> str:
        return get_settings().anthropic()[2]

    def configured(self) -> bool:
        return bool(get_settings().anthropic()[0])

    def chat(self, messages: list[dict], model: str | None = None) -> ChatResult:
        s = get_settings()
        key, base_url, default_model = s.anthropic()
        if not key:
            raise ProviderNotConfigured(
                "No Anthropic key. Set ANTHROPIC_API_KEY (or LFL_ANTHROPIC_API_KEY) in your .env."
            )
        model = model or default_model

        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        payload: dict = {"model": model, "max_tokens": s.max_output_tokens, "messages": convo}
        if system:
            payload["system"] = system

        start = time.perf_counter()
        r = post(
            f"{base_url}/v1/messages",
            json=payload,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            timeout=s.request_timeout_s,
            who="the Anthropic API",
        )
        reply = parse(r, _text_blocks, "Anthropic")
        return ChatResult(reply=reply, model=model, latency_ms=(time.perf_counter() - start) * 1000)
