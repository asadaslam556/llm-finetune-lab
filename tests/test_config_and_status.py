"""Settings behavior + the atomic status store."""

import json
import os
import threading

import pytest

from finetune_lab.core.config import get_settings
from finetune_lab.pipeline.status import STAGE_ORDER, StatusStore, _blank


class TestConfig:
    def test_defaults(self, settings):
        assert settings.default_provider == "ollama"
        assert settings.base_model_hf == "Qwen/Qwen2.5-0.5B-Instruct"
        assert settings.deploy_name == "nimbus-support"

    def test_env_override(self, env, monkeypatch):
        monkeypatch.setenv("LFL_DEFAULT_PROVIDER", "mock")
        monkeypatch.setenv("LFL_LORA_R", "8")
        monkeypatch.setenv("LFL_FINETUNE_STRATEGY", "lora")
        get_settings.cache_clear()
        s = get_settings()
        assert s.default_provider == "mock"
        assert s.lora_r == 8
        assert s.finetune_strategy == "lora"

    def test_qlora_is_the_default_strategy(self, settings):
        assert settings.finetune_strategy == "qlora"
        assert settings.quant_bits == 4
        assert settings.quant_type == "nf4"
        assert settings.double_quant is True

    def test_cors_origins_accept_a_comma_separated_string(self, env, monkeypatch):
        """Everybody types the comma form into a .env before reaching for
        JSON, and pydantic rejects it by default."""
        monkeypatch.setenv("LFL_CORS_ORIGINS", "http://a.test, http://b.test")
        get_settings.cache_clear()
        assert get_settings().cors_origins == ["http://a.test", "http://b.test"]

    def test_dirs_created(self, settings):
        assert settings.artifacts_dir.exists()
        assert settings.data_dir.exists()


class TestStatusStore:
    def test_read_missing_file_gives_idle(self, settings):
        store = StatusStore(settings.artifacts_dir / "status.json")
        status = store.read()
        assert status["state"] == "idle"
        assert set(status["stages"]) == set(STAGE_ORDER)

    def test_write_then_read_roundtrip(self, settings):
        store = StatusStore(settings.artifacts_dir / "status.json")
        status = store.start_run("run-1", True, STAGE_ORDER, settings.artifacts_dir / "run-1")
        assert store.read()["run_id"] == "run-1"
        store.stage_started(status, "ingest")
        assert store.read()["stages"]["ingest"]["state"] == "running"

    def test_stage_started_records_a_timestamp(self, settings):
        """The UI clock reads this; without it a reload restarts elapsed at 0."""
        store = StatusStore(settings.artifacts_dir / "status.json")
        status = store.start_run("r", True, STAGE_ORDER, settings.artifacts_dir / "r")
        store.stage_started(status, "ingest")
        assert store.read()["stages"]["ingest"]["started_at"] > 0

    def test_read_survives_a_locked_file(self, settings, monkeypatch):
        """Windows raises PermissionError when a writer swaps the file out
        from under a reader. That must not become a 500."""
        store = StatusStore(settings.artifacts_dir / "status.json")
        store.write(_blank("r"))

        real_open = open

        def locked(path, *a, **k):
            if str(path).endswith("status.json"):
                raise PermissionError(13, "used by another process")
            return real_open(path, *a, **k)

        monkeypatch.setattr("builtins.open", locked)
        assert store.read()["state"] == "idle"

    def test_write_retries_a_blocked_replace(self, settings, monkeypatch):
        """Same Windows situation from the writer's side: retry, don't die."""
        store = StatusStore(settings.artifacts_dir / "status.json")
        calls = {"n": 0}
        real_replace = os.replace

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError(13, "used by another process")
            return real_replace(src, dst)

        monkeypatch.setattr("finetune_lab.pipeline.status.os.replace", flaky)
        store.write(_blank("retried"))
        assert calls["n"] == 3
        assert store.read()["run_id"] == "retried"

    def test_write_gives_up_with_a_clear_message(self, settings, monkeypatch):
        store = StatusStore(settings.artifacts_dir / "status.json")

        def never(src, dst):
            raise PermissionError(13, "used by another process")

        monkeypatch.setattr("finetune_lab.pipeline.status.os.replace", never)
        with pytest.raises(OSError, match="holding it open"):
            store.write(_blank("nope"))

    def test_read_rejects_valid_json_of_the_wrong_shape(self, settings):
        path = settings.artifacts_dir / "status.json"
        path.write_text('["not", "a", "status"]', encoding="utf-8")
        assert StatusStore(path).read()["state"] == "idle"

    def test_corrupt_file_recovers_to_idle(self, settings):
        path = settings.artifacts_dir / "status.json"
        path.write_text("{definitely not json", encoding="utf-8")
        assert StatusStore(path).read()["state"] == "idle"

    def test_subset_marks_others_skipped(self):
        status = _blank("r", True, ["ingest", "prepare"])
        assert status["stages"]["finetune"]["state"] == "skipped"
        assert status["stages"]["ingest"]["state"] == "pending"

    def test_atomic_write_never_leaves_partial_reads(self, settings):
        """Hammer writes from a thread while reading; every read must parse."""
        store = StatusStore(settings.artifacts_dir / "status.json")
        base = _blank("r")
        base["stages"]["ingest"]["message"] = (
            "x" * 20000
        )  # big enough to make a torn write plausible
        stop = threading.Event()

        def writer():
            while not stop.is_set():
                store.write(base)

        t = threading.Thread(target=writer)
        t.start()
        try:
            for _ in range(200):
                json.dumps(store.read())  # raises if we ever see a torn file
        finally:
            stop.set()
            t.join()
