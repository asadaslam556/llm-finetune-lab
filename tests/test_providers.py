"""Provider layer: mock, registry, Ollama and the cloud APIs with faked httpx."""

import httpx
import pytest

import finetune_lab.providers.registry as registry
from finetune_lab.core.config import Settings, get_settings
from finetune_lab.providers.anthropic import AnthropicProvider
from finetune_lab.providers.base import ProviderNotConfigured, ProviderUnavailable
from finetune_lab.providers.mock import MockProvider
from finetune_lab.providers.ollama import OllamaProvider
from finetune_lab.providers.openai_compatible import (
    CustomProvider,
    DeepSeekProvider,
    OpenAIProvider,
)

MSGS = [{"role": "user", "content": "How do I pin a note?"}]


def set_env(monkeypatch, **pairs):
    """Set env vars and drop the settings cache, which is what actually makes
    them visible. Forgetting the second half is the classic flaky test here."""
    for k, v in pairs.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    registry.reset_cache()


def fake_post(handler):
    def post(url, json=None, headers=None, timeout=None):
        return handler(httpx.Request("POST", url, json=json))

    return post


def json_response(payload, status=200):
    return fake_post(lambda req: httpx.Response(status, json=payload, request=req))


class TestMock:
    def test_deterministic(self, env):
        p = MockProvider()
        assert p.chat(MSGS).reply == p.chat(MSGS).reply

    def test_different_questions_can_differ(self, env):
        p = MockProvider()
        replies = {
            p.chat([{"role": "user", "content": q}]).reply
            for q in ["pin?", "sync?", "share?", "trash?", "export?"]
        }
        assert len(replies) > 1  # not literally one canned string

    def test_always_configured(self, env):
        assert MockProvider().configured() is True


class TestRegistry:
    def test_known_names_resolve(self, env):
        for name in ("ollama", "anthropic", "deepseek", "openai", "custom", "mock"):
            assert registry.get_provider(name).name == name

    def test_unknown_name_raises_keyerror(self, env):
        with pytest.raises(KeyError, match="Unknown provider"):
            registry.get_provider("bard")

    def test_list_reports_configured_flags(self, env):
        info = {p["name"]: p for p in registry.list_providers()}
        assert info["mock"]["configured"] is True
        assert info["anthropic"]["configured"] is False
        assert info["deepseek"]["configured"] is False
        assert all("default_model" in p and "note" in p for p in info.values())

    def test_exactly_one_provider_is_marked_default(self, env, monkeypatch):
        """The console preselects this one. It used to hardcode ollama and
        ignore LFL_DEFAULT_PROVIDER entirely."""
        set_env(monkeypatch, LFL_DEFAULT_PROVIDER="mock")
        defaults = [p["name"] for p in registry.list_providers() if p["default"]]
        assert defaults == ["mock"]


class TestCredentialResolution:
    """Both naming styles have to work. Someone who already exports
    ANTHROPIC_API_KEY for the official SDK should not have to set it twice."""

    def test_vendor_standard_name_is_picked_up(self, env, monkeypatch):
        set_env(monkeypatch, ANTHROPIC_API_KEY="sk-ant-test")
        assert AnthropicProvider().configured() is True

    def test_namespaced_name_also_works(self, env, monkeypatch):
        set_env(monkeypatch, LFL_ANTHROPIC_API_KEY="sk-ant-test")
        assert AnthropicProvider().configured() is True

    def test_base_url_defaults_to_the_public_api(self, env):
        assert get_settings().anthropic()[1] == "https://api.anthropic.com"

    def test_gateway_base_url_overrides_it(self, env, monkeypatch):
        """This is what makes a corporate or self-hosted gateway work without
        a code change. Trailing slashes get stripped so URL joining is safe."""
        set_env(monkeypatch, ANTHROPIC_BASE_URL="https://gateway.example.com/")
        assert get_settings().anthropic()[1] == "https://gateway.example.com"

    def test_model_override(self, env, monkeypatch):
        set_env(monkeypatch, ANTHROPIC_MODEL="claude-sonnet-5@default")
        assert AnthropicProvider().default_model() == "claude-sonnet-5@default"

    def test_hf_token_falls_back_to_the_standard_var(self, env, monkeypatch):
        set_env(monkeypatch, HF_TOKEN="hf_abc123")
        assert get_settings().resolved_hf_token() == "hf_abc123"

    def test_vendor_name_in_a_dotenv_file_is_picked_up(self, env, monkeypatch):
        """Regression: pydantic only reads LFL_* names out of .env, so a plain
        ANTHROPIC_API_KEY line there used to be silently ignored."""
        dotenv = env / ".env"
        dotenv.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n", encoding="utf-8")
        monkeypatch.setitem(Settings.model_config, "env_file", str(dotenv))
        set_env(monkeypatch)
        assert get_settings().anthropic()[0] == "sk-ant-from-file"


class TestAnthropic:
    def test_posts_to_the_configured_base_url(self, env, monkeypatch):
        set_env(
            monkeypatch,
            ANTHROPIC_API_KEY="sk-ant-test",
            ANTHROPIC_BASE_URL="https://gateway.example.com",
        )
        seen = {}

        def post(url, json=None, headers=None, timeout=None):
            seen["url"] = url
            seen["headers"] = headers
            seen["json"] = json
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "Long-press and choose Pin."}]},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr("finetune_lab.providers.http.httpx.post", post)
        result = AnthropicProvider().chat(MSGS)
        assert seen["url"] == "https://gateway.example.com/v1/messages"
        assert seen["headers"]["x-api-key"] == "sk-ant-test"
        assert "Pin" in result.reply

    def test_system_prompt_goes_top_level_not_in_messages(self, env, monkeypatch):
        """The one real difference from the OpenAI shape. Leaving a system
        message in the array gets a 400 from the API."""
        set_env(monkeypatch, ANTHROPIC_API_KEY="sk-ant-test")
        seen = {}

        def post(url, json=None, headers=None, timeout=None):
            seen.update(json or {})
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "ok"}]},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr("finetune_lab.providers.http.httpx.post", post)
        AnthropicProvider().chat([{"role": "system", "content": "Be terse."}, *MSGS])
        assert seen["system"] == "Be terse."
        assert all(m["role"] != "system" for m in seen["messages"])

    def test_without_key_raises_not_configured(self, env):
        with pytest.raises(ProviderNotConfigured, match="ANTHROPIC_API_KEY"):
            AnthropicProvider().chat(MSGS)


class TestErrorsNameTheServer:
    def test_http_error_says_which_host_answered(self, env, monkeypatch):
        """A base URL from the shell beats the one in .env. When that sends a
        gateway key to the wrong server, the error has to say which server."""
        set_env(monkeypatch, ANTHROPIC_API_KEY="k", ANTHROPIC_BASE_URL="https://gw.example.com")
        monkeypatch.setattr(
            "finetune_lab.providers.http.httpx.post",
            json_response({"error": {"message": "invalid x-api-key"}}, status=401),
        )
        with pytest.raises(ProviderUnavailable, match=r"at gw\.example\.com returned HTTP 401"):
            AnthropicProvider().chat(MSGS)


class TestOpenAICompatible:
    def test_deepseek_hits_its_own_host(self, env, monkeypatch):
        set_env(monkeypatch, DEEPSEEK_API_KEY="sk-ds-test")
        seen = {}

        def post(url, json=None, headers=None, timeout=None):
            seen["url"] = url
            seen["model"] = (json or {}).get("model")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Deleted notes sit in Trash."}}]},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr("finetune_lab.providers.http.httpx.post", post)
        result = DeepSeekProvider().chat(MSGS)
        assert seen["url"] == "https://api.deepseek.com/v1/chat/completions"
        assert seen["model"] == "deepseek-chat"
        assert "Trash" in result.reply

    def test_custom_provider_needs_all_three_vars(self, env, monkeypatch):
        assert CustomProvider().configured() is False
        set_env(monkeypatch, CUSTOM_API_KEY="k", CUSTOM_BASE_URL="https://x/v1")
        assert CustomProvider().configured() is False  # still no model
        set_env(monkeypatch, CUSTOM_MODEL="some-model")
        assert CustomProvider().configured() is True

    def test_custom_provider_describes_where_it_points(self, env, monkeypatch):
        set_env(
            monkeypatch,
            CUSTOM_API_KEY="k",
            CUSTOM_BASE_URL="https://api.together.xyz/v1",
            CUSTOM_MODEL="llama-3.3-70b",
            LFL_CUSTOM_LABEL="Together",
        )
        note = CustomProvider().describe()
        assert "Together" in note and "together.xyz" in note

    def test_openai_without_key_raises_not_configured(self, env):
        with pytest.raises(ProviderNotConfigured, match="OPENAI_API_KEY"):
            OpenAIProvider().chat(MSGS)


class TestOllama:
    def test_parses_chat_response(self, env, monkeypatch):
        monkeypatch.setattr(
            "finetune_lab.providers.ollama.httpx.post",
            json_response({"message": {"role": "assistant", "content": "Long-press and Pin."}}),
        )
        result = OllamaProvider().chat(MSGS, model="qwen2.5:0.5b-instruct")
        assert "Pin" in result.reply
        assert result.latency_ms >= 0

    def test_connection_error_is_readable(self, env, monkeypatch):
        def post(url, json=None, timeout=None):
            raise httpx.ConnectError("boom")

        monkeypatch.setattr("finetune_lab.providers.ollama.httpx.post", post)
        with pytest.raises(ProviderUnavailable, match="ollama serve"):
            OllamaProvider().chat(MSGS)

    def test_unknown_model_404_suggests_pull(self, env, monkeypatch):
        monkeypatch.setattr(
            "finetune_lab.providers.ollama.httpx.post",
            fake_post(lambda req: httpx.Response(404, text="model not found", request=req)),
        )
        with pytest.raises(ProviderUnavailable, match="ollama pull"):
            OllamaProvider().chat(MSGS, model="nope:latest")

    def test_default_model_flips_after_deploy_marker(self, env, settings):
        p = OllamaProvider()
        assert p.default_model() == settings.ollama_base_tag
        (settings.artifacts_dir / "deployed.txt").write_text("nimbus-support", encoding="utf-8")
        assert p.default_model() == "nimbus-support"


class TestParsingFailures:
    """A 200 with an unreadable body still means we cannot answer. It has to
    come back as a ProviderError so the chat panel prints a sentence rather
    than the API handing the browser a 500."""

    def test_ollama_html_body_is_an_outage(self, env, monkeypatch):
        def post(url, json=None, timeout=None):
            return httpx.Response(
                200, text="<html>502 Bad Gateway</html>", request=httpx.Request("POST", url)
            )

        monkeypatch.setattr("finetune_lab.providers.ollama.httpx.post", post)
        with pytest.raises(ProviderUnavailable, match="not JSON"):
            OllamaProvider().chat(MSGS)

    def test_ollama_json_without_a_message_is_an_outage(self, env, monkeypatch):
        monkeypatch.setattr(
            "finetune_lab.providers.ollama.httpx.post", json_response({"unexpected": True})
        )
        with pytest.raises(ProviderUnavailable):
            OllamaProvider().chat(MSGS)

    def test_openai_error_shaped_200_is_an_outage(self, env, monkeypatch):
        """Gateways love returning their own error envelope with a 200."""
        set_env(monkeypatch, OPENAI_API_KEY="sk-test")
        monkeypatch.setattr(
            "finetune_lab.providers.http.httpx.post",
            json_response({"error": {"message": "quota exceeded"}}),
        )
        with pytest.raises(ProviderUnavailable, match="could not read"):
            OpenAIProvider().chat(MSGS)

    def test_api_error_message_is_pulled_out_of_the_body(self, env, monkeypatch):
        set_env(monkeypatch, OPENAI_API_KEY="sk-test")
        monkeypatch.setattr(
            "finetune_lab.providers.http.httpx.post",
            json_response({"error": {"message": "Rate limit reached for gpt-4o-mini"}}, status=429),
        )
        with pytest.raises(ProviderUnavailable, match="Rate limit reached"):
            OpenAIProvider().chat(MSGS)

    def test_empty_reply_is_an_outage_not_an_empty_bubble(self, env, monkeypatch):
        set_env(monkeypatch, OPENAI_API_KEY="sk-test")
        monkeypatch.setattr(
            "finetune_lab.providers.http.httpx.post",
            json_response({"choices": [{"message": {"content": "   "}}]}),
        )
        with pytest.raises(ProviderUnavailable, match="empty reply"):
            OpenAIProvider().chat(MSGS)
