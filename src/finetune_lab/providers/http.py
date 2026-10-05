"""Shared HTTP helpers for the cloud providers.

Raw httpx instead of the official SDKs, on purpose: two fewer dependencies,
and the request shape is small enough that an SDK buys us nothing. What it
does cost us is error handling, which is what this module is.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from .base import ProviderUnavailable


def api_error(r: httpx.Response) -> str:
    """Pull the useful sentence out of an error body if there is one.

    Anthropic and the OpenAI-shaped APIs both nest it under "error", and
    their raw JSON is noisy enough that dumping it into a chat bubble helps
    nobody.
    """
    try:
        err = r.json().get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:300]
        if isinstance(err, str):
            return err[:300]
    except ValueError:
        pass
    return r.text[:200]


def parse(r: httpx.Response, extract, who: str) -> str:
    """Run an extractor over a response body, turning any surprise in its
    shape into a normal provider outage.

    A 200 with an unexpected payload still means we cannot answer, and that
    should read like a sentence rather than a 500. Gateways in front of these
    APIs are the usual cause: they happily return HTML or their own error
    envelope with a 200.
    """
    try:
        body = r.json()
        value = extract(body)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise ProviderUnavailable(
            f"{who} returned a response we could not read ({type(e).__name__}). "
            f"Body starts: {r.text[:160]}"
        ) from e
    if not isinstance(value, str) or not value.strip():
        if _ran_out_of_tokens(body):
            raise ProviderUnavailable(
                f"{who} used the whole reply budget (LFL_MAX_OUTPUT_TOKENS) before writing an "
                "answer. Reasoning models spend it on hidden thinking first: raise "
                "LFL_MAX_OUTPUT_TOKENS, or use a non-reasoning model such as deepseek-chat."
            )
        raise ProviderUnavailable(
            f"{who} returned an empty reply. That usually means the model ran but produced "
            "nothing. Try a different prompt, or check the model name is right for this endpoint."
        )
    return value


def _ran_out_of_tokens(body) -> bool:
    """OpenAI-shaped APIs say finish_reason "length", Anthropic says stop_reason "max_tokens"."""
    if not isinstance(body, dict):
        return False
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else {}
    return (isinstance(first, dict) and first.get("finish_reason") == "length") or (
        body.get("stop_reason") == "max_tokens"
    )


def post(url: str, *, json: dict, headers: dict, timeout: float, who: str) -> httpx.Response:
    """POST and normalise the two failure shapes into ProviderUnavailable."""
    try:
        r = httpx.post(url, json=json, headers=headers, timeout=timeout)
    except httpx.HTTPError as e:
        raise ProviderUnavailable(
            f"Could not reach {who} at {url} ({type(e).__name__}). Check the base URL, "
            "your network, and any proxy settings."
        ) from e
    if r.status_code >= 400:
        # Name the host: a base URL inherited from the shell silently beats the
        # one in .env, and "401 at api.anthropic.com" makes that obvious where
        # a bare "401" sends you off to regenerate a perfectly good key.
        host = urlparse(url).hostname or url
        raise ProviderUnavailable(f"{who} at {host} returned HTTP {r.status_code}: {api_error(r)}")
    return r
