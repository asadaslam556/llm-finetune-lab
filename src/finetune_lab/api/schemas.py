"""Request and response shapes for the API.

Kept in one file on purpose. This project is small enough that hunting across
modules for a schema is worse than one slightly long file.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    # Bounded because chat is forwarded to paid APIs. Generous for a person
    # typing, far too small for anyone trying to run up a bill.
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    provider: str | None = Field(default=None, max_length=64)  # else settings.default_provider
    model: str | None = Field(default=None, max_length=200)  # else the provider's default


class ChatResponse(BaseModel):
    ok: bool
    provider: str
    model: str
    reply: str
    latency_ms: float
    # ok=False means the provider was down or misconfigured, and reply then
    # holds a human-readable explanation instead of model output. Deliberate:
    # a dead Ollama should read like a sentence in the UI, not a 500.
    error_kind: str | None = None


class ProviderInfo(BaseModel):
    name: str
    configured: bool
    default_model: str
    note: str
    # True for the one LFL_DEFAULT_PROVIDER names, so the console can preselect it.
    default: bool = False


class PipelineRunRequest(BaseModel):
    dry_run: bool = True
    # A subset like ["ingest", "prepare"] re-runs part of the pipeline.
    # Empty or None means all seven stages in order.
    stages: list[str] | None = Field(default=None, max_length=20)


class PipelineRunAccepted(BaseModel):
    run_id: str
    dry_run: bool
    stages: list[str]
    run_dir: str
