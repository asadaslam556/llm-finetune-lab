"""Provider abstraction.

Every chat backend implements the same tiny interface:
`chat(messages, model) -> ChatResult`. The app talks to the interface only,
so swapping providers is an env var rather than a refactor.

Worth being clear about what this layer is for: these are *inference*
providers used to talk to a model and to A/B your fine-tune against a hosted
one. None of them train anything. Training happens in the pipeline against
open-weight models you download, because no hosted chat API exposes a QLoRA
fine-tuning endpoint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class ProviderError(Exception):
    """Base for anything a provider can throw at us."""

    kind = "provider_error"


class ProviderNotConfigured(ProviderError):
    """Provider exists but is missing what it needs, usually an API key."""

    kind = "not_configured"


class ProviderUnavailable(ProviderError):
    """We tried to reach it and could not. Network down, Ollama not running,
    model tag not pulled, that sort of thing."""

    kind = "unavailable"


@dataclass
class ChatResult:
    reply: str
    model: str
    latency_ms: float


class Provider(ABC):
    name: str = "base"
    note: str = ""

    @abstractmethod
    def chat(self, messages: list[dict], model: str | None = None) -> ChatResult:
        """Send the conversation, get one assistant reply back.

        Blocking on purpose. FastAPI runs sync endpoints in a threadpool, and
        blocking httpx calls are far easier to fake in tests than async ones.
        """

    @abstractmethod
    def default_model(self) -> str: ...

    def configured(self) -> bool:
        """Can this provider plausibly work right now?

        Cheap check only, no network. The providers list endpoint hits this
        on every page load.
        """
        return True

    def describe(self) -> str:
        """One line for the provider picker. Override it when the useful
        description depends on config rather than being a constant."""
        return self.note
