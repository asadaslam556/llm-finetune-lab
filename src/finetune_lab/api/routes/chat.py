"""Chat endpoint.

The one deliberate quirk: provider failures come back as HTTP 200 with
ok=False and a readable sentence in `reply`. A dead Ollama is a normal state
of the world for this app, because you probably just have not started it yet,
so the UI shows "Could not reach Ollama..." in the chat transcript instead of
a red toast and a stack trace.

Unknown provider names are still a hard 400. That is a caller bug, not
weather.
"""

from fastapi import APIRouter, HTTPException

from ...core.config import get_settings
from ...core.logging import log
from ...providers.base import ProviderError
from ...providers.registry import get_provider
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    name = req.provider or get_settings().default_provider
    try:
        provider = get_provider(name)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    messages = [m.model_dump() for m in req.messages]
    try:
        result = provider.chat(messages, model=req.model)
    except ProviderError as e:
        log.warning("provider %s failed: %s", name, e)
        return ChatResponse(
            ok=False,
            provider=name,
            model=req.model or provider.default_model(),
            reply=str(e),
            latency_ms=0.0,
            error_kind=e.kind,
        )
    return ChatResponse(
        ok=True,
        provider=name,
        model=result.model,
        reply=result.reply,
        latency_ms=round(result.latency_ms, 1),
    )
