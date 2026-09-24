"""Shared fixtures.

Every test gets its own artifacts/ and data/ under tmp_path, wired in via
LFL_* env vars, with the settings cache and the provider cache cleared so
nothing leaks between tests. The seed dataset is copied in, so stage tests run
against the same data a human sees on first clone.

Provider credentials get scrubbed too. Otherwise a real .env on the developer's
machine changes what the provider tests see, and the suite passes for you and
fails in CI.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import finetune_lab.pipeline.runner as runner
import finetune_lab.providers.registry as registry
from finetune_lab.core.config import Settings, get_settings

REPO = Path(__file__).resolve().parents[1]

# Anything that could leak a real credential into a test run.
CREDENTIAL_VARS = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "CUSTOM_API_KEY",
    "CUSTOM_BASE_URL",
    "CUSTOM_MODEL",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    shutil.copy(REPO / "data" / "raw" / "nimbus_tickets.jsonl", data / "raw")

    for var in CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(f"LFL_{var}", raising=False)
    # A real .env in the repo root would otherwise leak into every test, both
    # through pydantic and through the vendor-name fallback in config._env.
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("LFL_DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("LFL_DATA_DIR", str(data))
    monkeypatch.setenv("LFL_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    # TestClient sends Host: testserver.
    monkeypatch.setenv("LFL_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver")

    get_settings.cache_clear()
    registry.reset_cache()
    runner._set_running(False)  # a leak in one test should not 409 the next one
    yield tmp_path
    get_settings.cache_clear()
    registry.reset_cache()
    runner._set_running(False)


@pytest.fixture
def settings(env):
    return get_settings()


@pytest.fixture
def run_dir(settings):
    d = settings.artifacts_dir / "test-run"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def client(env):
    from fastapi.testclient import TestClient

    from finetune_lab.api.app import create_app

    return TestClient(create_app(), raise_server_exceptions=False)
