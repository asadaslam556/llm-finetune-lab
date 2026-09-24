"""Provider registry.

Central name to provider lookup. Adding a provider is: write the class, add
one line to _FACTORIES, done. Instances are cached because providers hold no
state beyond config.
"""

from __future__ import annotations

from ..core.config import get_settings
from .anthropic import AnthropicProvider
from .base import Provider
from .mock import MockProvider
from .ollama import OllamaProvider
from .openai_compatible import CustomProvider, DeepSeekProvider, OpenAIProvider

_FACTORIES: dict[str, type[Provider]] = {
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
    "deepseek": DeepSeekProvider,
    "openai": OpenAIProvider,
    "custom": CustomProvider,
    "mock": MockProvider,
}

_instances: dict[str, Provider] = {}


def known_providers() -> list[str]:
    return list(_FACTORIES)


def get_provider(name: str) -> Provider:
    if name not in _FACTORIES:
        raise KeyError(f"Unknown provider '{name}'. Known: {', '.join(sorted(_FACTORIES))}")
    if name not in _instances:
        _instances[name] = _FACTORIES[name]()
    return _instances[name]


def list_providers() -> list[dict]:
    default = get_settings().default_provider
    out = []
    for name in _FACTORIES:
        p = get_provider(name)
        out.append(
            {
                "name": name,
                "configured": p.configured(),
                "default_model": p.default_model(),
                "note": p.describe(),
                "default": name == default,
            }
        )
    return out


def reset_cache() -> None:
    """Tests poke at env vars, so they need a way to drop cached instances."""
    _instances.clear()
