"""The health endpoint, which is only worth having if it actually probes.

The thing being pinned down here is that /api/health reflects reality: it
writes to storage for real, opens a socket to Ollama for real, and reports
what it found instead of a cheerful constant.
"""

import socket

import httpx
import pytest

from finetune_lab.api.routes import health

# Not a real token. Long enough to exercise masking.
FAKE_HF_TOKEN = "hf_abcdefghijklmnopqrstuvwxyz123456"  # gitleaks:allow


class TestLiveness:
    def test_live_is_a_constant_on_purpose(self, client):
        r = client.get("/api/health/live")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_live_touches_no_dependencies(self, client, monkeypatch):
        """Liveness must answer even with every dependency on fire."""

        def explode():
            raise RuntimeError("everything is broken")

        monkeypatch.setattr(health, "_check_artifacts_writable", explode)
        monkeypatch.setattr(health, "_check_ollama", explode)
        assert client.get("/api/health/live").status_code == 200


class TestHealthProbes:
    def test_reports_one_result_per_dependency(self, client):
        body = client.get("/api/health").json()
        names = {c["name"] for c in body["checks"]}
        assert names == {
            "storage.artifacts",
            "storage.dataset",
            "storage.status_file",
            "providers.config",
            "providers.ollama",
            "providers.huggingface",
            "training.extras",
            "training.quantization",
        }
        assert all("detail" in c and "ms" in c for c in body["checks"])

    def test_not_a_static_ok(self, client):
        """No Ollama running in the test environment, so an honest health
        check cannot come back clean."""
        body = client.get("/api/health").json()
        assert body["status"] in ("degraded", "error")
        assert "providers.ollama" in body["failing"]

    def test_storage_check_actually_writes(self, client, settings, monkeypatch):
        """Point the probe at a directory it can't write to and it must fail,
        which proves it's doing more than checking that a path exists."""

        def readonly(*a, **k):
            raise PermissionError(13, "read-only file system")

        monkeypatch.setattr("pathlib.Path.write_text", readonly)
        r = client.get("/api/health")
        assert r.status_code == 503
        body = r.json()
        assert body["status"] == "error"
        assert "storage.artifacts" in body["failing"]

    def test_missing_dataset_degrades_but_stays_usable(self, client, settings):
        for f in (settings.data_dir / "raw").glob("*.jsonl"):
            f.unlink()
        r = client.get("/api/health")
        assert r.status_code == 200  # you can still chat; only training is blocked
        body = r.json()
        assert body["status"] == "degraded"
        check = next(c for c in body["checks"] if c["name"] == "storage.dataset")
        assert "make_seed_data" in check["detail"]

    def test_one_broken_probe_does_not_take_down_the_endpoint(self, client, monkeypatch):
        def explode():
            raise ValueError("probe itself is buggy")

        monkeypatch.setattr(health, "_check_training_extras", explode)
        r = client.get("/api/health")
        assert r.status_code == 200
        check = next(c for c in r.json()["checks"] if c["name"] == "training.extras")
        assert check["ok"] is False
        assert "ValueError" in check["detail"]

    def test_ollama_probe_distinguishes_dead_port_from_wrong_service(self, client, monkeypatch):
        # nothing listening
        def refuse(*a, **k):
            raise ConnectionRefusedError("nope")

        monkeypatch.setattr(socket, "create_connection", refuse)
        detail = next(
            c for c in client.get("/api/health").json()["checks"] if c["name"] == "providers.ollama"
        )["detail"]
        assert "nothing listening" in detail
        assert "ollama serve" in detail

        # something listening, but it isn't Ollama
        class FakeSock:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: FakeSock())
        monkeypatch.setattr(
            health.httpx,
            "get",
            lambda url, timeout=None: httpx.Response(404, request=httpx.Request("GET", url)),
        )
        detail = next(
            c for c in client.get("/api/health").json()["checks"] if c["name"] == "providers.ollama"
        )["detail"]
        assert "really Ollama" in detail

    def test_ollama_up_but_model_not_pulled_is_reported(self, client, monkeypatch):
        class FakeSock:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: FakeSock())
        monkeypatch.setattr(
            health.httpx,
            "get",
            lambda url, timeout=None: httpx.Response(
                200, json={"models": [{"name": "llama3:8b"}]}, request=httpx.Request("GET", url)
            ),
        )
        detail = next(
            c for c in client.get("/api/health").json()["checks"] if c["name"] == "providers.ollama"
        )["detail"]
        assert "is not pulled" in detail
        assert "ollama pull" in detail

    def test_ollama_healthy_when_the_tag_is_present(self, client, monkeypatch, settings):
        class FakeSock:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: FakeSock())
        monkeypatch.setattr(
            health.httpx,
            "get",
            lambda url, timeout=None: httpx.Response(
                200,
                json={"models": [{"name": settings.ollama_base_tag}]},
                request=httpx.Request("GET", url),
            ),
        )
        check = next(
            c for c in client.get("/api/health").json()["checks"] if c["name"] == "providers.ollama"
        )
        assert check["ok"] is True

    def test_unknown_default_provider_is_a_required_failure(self, client, env, monkeypatch):
        monkeypatch.setenv("LFL_DEFAULT_PROVIDER", "telepathy")
        from finetune_lab.core.config import get_settings

        get_settings.cache_clear()
        r = client.get("/api/health")
        assert r.status_code == 503
        assert "providers.config" in r.json()["failing"]

    def test_hf_auth_missing_token_is_not_a_failure(self, client):
        """Public models need no token, so absence must not show as broken."""
        check = next(
            c
            for c in client.get("/api/health").json()["checks"]
            if c["name"] == "providers.huggingface"
        )
        assert check["ok"] is True
        assert "public models" in check["detail"]

    def test_hf_token_is_masked_in_the_response(self, client, env, monkeypatch):
        """Health output gets pasted into issues and screenshots."""
        monkeypatch.setenv("LFL_HF_TOKEN", FAKE_HF_TOKEN)
        from finetune_lab.core.config import get_settings

        get_settings.cache_clear()
        detail = next(
            c
            for c in client.get("/api/health").json()["checks"]
            if c["name"] == "providers.huggingface"
        )["detail"]
        assert FAKE_HF_TOKEN not in detail
        assert "hf_abc" in detail and "3456" in detail

    def test_malformed_hf_token_is_flagged(self, client, env, monkeypatch):
        monkeypatch.setenv("LFL_HF_TOKEN", "sk-wrong-provider-entirely")
        from finetune_lab.core.config import get_settings

        get_settings.cache_clear()
        check = next(
            c
            for c in client.get("/api/health").json()["checks"]
            if c["name"] == "providers.huggingface"
        )
        assert check["ok"] is False

    def test_reports_live_pipeline_state(self, client):
        assert client.get("/api/health").json()["pipeline_busy"] is False

    @pytest.mark.parametrize(
        "field", ["default_provider", "base_model", "deploy_name", "ollama_host"]
    )
    def test_config_echo(self, client, field):
        assert field in client.get("/api/health").json()["config"]

    def test_health_does_not_drag_torch_into_the_api_process(self, client, monkeypatch):
        """Regression: this check used to `import torch` for GPU details,
        which cost 1.5s on the first call and pulled roughly a gigabyte into
        a process that's supposed to stay free of the training stack."""
        import builtins

        real_import = builtins.__import__
        pulled = []

        def watch(name, *a, **k):
            if name.split(".")[0] in ("torch", "transformers", "peft", "datasets"):
                pulled.append(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", watch)
        client.get("/api/health")
        assert pulled == [], f"health imported heavy modules: {pulled}"

    def test_extras_check_is_fast(self, client):
        check = next(
            c for c in client.get("/api/health").json()["checks"] if c["name"] == "training.extras"
        )
        assert check["ms"] < 250, "probe is slow enough to annoy a poller"
