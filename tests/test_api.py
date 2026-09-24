"""End-to-end through FastAPI's TestClient."""

import time

import pytest

import finetune_lab.pipeline.runner as runner


def wait_done(client, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/api/pipeline/status").json()
        if status["state"] in ("done", "failed") and not runner.is_running():
            return status
        time.sleep(0.05)
    raise TimeoutError


class TestBasics:
    def test_health_reports_real_dependency_state(self, client):
        # detailed coverage lives in test_health.py; this just proves it's wired up
        body = client.get("/api/health").json()
        assert body["status"] in ("ok", "degraded", "error")
        assert len(body["checks"]) == 8

    def test_providers_list(self, client):
        names = {p["name"] for p in client.get("/api/providers").json()}
        assert names == {"ollama", "anthropic", "deepseek", "openai", "custom", "mock"}

    def test_root_redirects_to_the_api_docs(self, client):
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"] == "/docs"

    def test_a_plain_caller_request_id_is_kept(self, client):
        r = client.get("/api/health/live", headers={"X-Request-ID": "trace-42.a_b"})
        assert r.headers["X-Request-ID"] == "trace-42.a_b"

    @pytest.mark.parametrize("bad", ["x" * 65, "has space", "semi;colon", "<script>"])
    def test_an_unsafe_caller_request_id_is_replaced(self, client, bad):
        r = client.get("/api/health/live", headers={"X-Request-ID": bad})
        assert r.headers["X-Request-ID"] != bad
        assert len(r.headers["X-Request-ID"]) == 12

    def test_request_id_and_timing_headers_present(self, client):
        r = client.get("/api/health/live")
        assert r.headers.get("X-Request-ID")
        assert float(r.headers["X-Response-Time-Ms"]) >= 0

    def test_error_responses_keep_their_cors_headers(self, client):
        """Regression: the tracing middleware used to sit outside CORS, so a
        500 reached the browser with no Access-Control headers and the fetch
        failed opaquely. The carefully written error message never arrived."""
        from finetune_lab.api.app import create_app

        app = create_app()

        @app.get("/api/_explode")
        def explode():
            raise RuntimeError("boom")

        from fastapi.testclient import TestClient

        c = TestClient(app, raise_server_exceptions=False)
        r = c.get("/api/_explode", headers={"Origin": "http://localhost:5173"})
        assert r.status_code == 500
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert r.headers.get("X-Request-ID")

    def test_foreign_host_header_is_refused(self, client):
        """DNS rebinding: an attacker's page keeps its own Host header."""
        r = client.get("/api/health/live", headers={"Host": "evil.example"})
        assert r.status_code == 400

    def test_localhost_with_a_port_is_allowed(self, client):
        r = client.get("/api/health/live", headers={"Host": "localhost:8000"})
        assert r.status_code == 200

    def test_tracing_headers_are_readable_by_the_browser(self, client):
        r = client.get("/api/health/live", headers={"Origin": "http://localhost:5173"})
        exposed = r.headers.get("access-control-expose-headers", "")
        assert "X-Request-ID" in exposed


class TestChat:
    def test_chat_with_mock_provider(self, client):
        r = client.post(
            "/api/chat",
            json={
                "provider": "mock",
                "messages": [{"role": "user", "content": "How do I pin a note?"}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["provider"] == "mock"
        assert len(body["reply"]) > 0

    def test_unknown_provider_is_400(self, client):
        r = client.post(
            "/api/chat",
            json={
                "provider": "bard",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 400
        assert "Unknown provider" in r.json()["detail"]

    def test_unconfigured_cloud_provider_degrades_readably(self, client):
        r = client.post(
            "/api/chat",
            json={
                "provider": "anthropic",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200  # outages are weather, not server errors
        body = r.json()
        assert body["ok"] is False
        assert body["error_kind"] == "not_configured"
        assert "ANTHROPIC_API_KEY" in body["reply"]

    def test_empty_messages_is_422(self, client):
        assert client.post("/api/chat", json={"messages": []}).status_code == 422

    def test_too_many_messages_is_422(self, client):
        msgs = [{"role": "user", "content": "hi"}] * 101
        r = client.post("/api/chat", json={"provider": "mock", "messages": msgs})
        assert r.status_code == 422

    def test_oversized_message_is_422(self, client):
        big = [{"role": "user", "content": "x" * 20_001}]
        r = client.post("/api/chat", json={"provider": "mock", "messages": big})
        assert r.status_code == 422

    def test_limits_leave_normal_use_alone(self, client):
        msgs = [{"role": "user", "content": "x" * 20_000}] * 100
        r = client.post("/api/chat", json={"provider": "mock", "messages": msgs})
        assert r.status_code == 200


class TestPipelineEndpoints:
    def test_status_starts_idle(self, client):
        body = client.get("/api/pipeline/status").json()
        assert body["state"] == "idle"
        assert body["busy"] is False

    def test_status_exposes_live_lock_state(self, client, monkeypatch):
        """The UI disables Start on this. It comes from the lock, not the
        file, because the file goes to done a moment before the lock frees."""

        def slow(ctx):
            time.sleep(0.8)
            return "slept", {}

        monkeypatch.setitem(runner.STAGE_FUNCS, "ingest", slow)
        client.post("/api/pipeline/run", json={"dry_run": True, "stages": ["ingest"]})
        assert client.get("/api/pipeline/status").json()["busy"] is True
        wait_done(client)
        assert client.get("/api/pipeline/status").json()["busy"] is False

    def test_subset_without_a_prior_run_is_400_not_a_dead_run(self, client):
        r = client.post("/api/pipeline/run", json={"dry_run": True, "stages": ["evaluate"]})
        assert r.status_code == 400
        assert "is not one yet" in r.json()["detail"]

    def test_unknown_stage_name_is_400(self, client):
        r = client.post("/api/pipeline/run", json={"dry_run": True, "stages": ["compile_kernel"]})
        assert r.status_code == 400
        assert "Unknown stage" in r.json()["detail"]

    def test_dry_run_end_to_end(self, client):
        r = client.post("/api/pipeline/run", json={"dry_run": True})
        assert r.status_code == 202
        run_id = r.json()["run_id"]
        status = wait_done(client)
        assert status["state"] == "done"
        assert status["run_id"] == run_id
        assert status["dry_run"] is True

    def test_busy_run_is_409(self, client, monkeypatch):
        def slow(ctx):
            time.sleep(1.0)
            return "slept", {}

        monkeypatch.setitem(runner.STAGE_FUNCS, "ingest", slow)
        assert (
            client.post(
                "/api/pipeline/run", json={"dry_run": True, "stages": ["ingest"]}
            ).status_code
            == 202
        )
        assert client.post("/api/pipeline/run", json={"dry_run": True}).status_code == 409
        wait_done(client)
