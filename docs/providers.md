# Providers

![DeepSeek](https://img.shields.io/badge/DeepSeek-API-4D6BFE?logo=deepseek&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-191919?logo=anthropic&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991)
![Ollama](https://img.shields.io/badge/Ollama-local-000000?logo=ollama&logoColor=white)
![HTTPX](https://img.shields.io/badge/HTTPX-client-2B5B84)
![dotenv](https://img.shields.io/badge/.env-gitignored-ECD53F?logo=dotenv&logoColor=black)

The chat panel talks to a model through a provider. Switching provider is a `.env` edit.

---

## What providers are for

> [!IMPORTANT]
> Providers are for **chatting with** models, not training them. None of these hosted APIs expose a QLoRA or LoRA fine-tuning endpoint for their chat models. Training always happens in the pipeline, on an open-weight model from Hugging Face.

![Chat providers behind one interface: the chat panel posts to /api/chat, the registry picks OllamaProvider, AnthropicProvider, OpenAICompatible (OpenAI, DeepSeek, custom) or MockProvider](images/providers.svg)

Every provider implements `chat(messages) -> ChatResult`, and `providers/registry.py` maps a name to one. DeepSeek, OpenAI and any custom endpoint share one class, `OpenAICompatible`, because they speak the same `/chat/completions` API.

The typical loop: turn on **Compare** and ask the same Nimbus question to your fine-tune and to DeepSeek or Claude. You see how close a 1.5B model trained on about 110 tickets gets to a frontier model, and where it beats one that has never heard of Nimbus.

![Chat in Compare mode: the console sends two POST /api/chat requests at once; the API looks up each provider in the registry, calls Ollama and Anthropic, and returns ok, reply and latency for each](images/chat-compare.svg)

---

## Setup per provider

Put these in `.env` (copied from `.env.example`), then restart the API.

### DeepSeek

```env
LFL_DEFAULT_PROVIDER=deepseek
DEEPSEEK_API_KEY=<YOUR_API_KEY>
DEEPSEEK_MODEL=deepseek-chat
```

Keys come from [platform.deepseek.com](https://platform.deepseek.com). `DEEPSEEK_BASE_URL` defaults to `https://api.deepseek.com/v1`, and `deepseek-chat` is the default model.

> [!NOTE]
> Reasoning models such as `deepseek-flash` think before they answer, and the thinking counts against `LFL_MAX_OUTPUT_TOKENS` (default 1024). If it runs out, the chat says so instead of showing an empty reply. `deepseek-chat` answers directly and fits the default; for a reasoning model, set `LFL_MAX_OUTPUT_TOKENS=4096`.

### Anthropic (Claude)

```env
LFL_DEFAULT_PROVIDER=anthropic
ANTHROPIC_API_KEY=<YOUR_API_KEY>
ANTHROPIC_MODEL=claude-sonnet-5
```

To route through a gateway (a corporate proxy, LiteLLM, or any Anthropic-compatible endpoint), add the base URL. Requests go to `{base}/v1/messages` with the same body and headers:

```env
ANTHROPIC_BASE_URL=https://your-gateway.example.com
ANTHROPIC_MODEL=claude-sonnet-5@default
```

Model names with gateway suffixes like `@default` are passed through untouched.

### OpenAI

```env
LFL_DEFAULT_PROVIDER=openai
OPENAI_API_KEY=<YOUR_API_KEY>
OPENAI_MODEL=gpt-4o-mini
```

### Anything else OpenAI-compatible

Together, Groq, OpenRouter, Fireworks, a local vLLM or LM Studio server. Three variables and the `custom` provider appears in the picker:

```env
LFL_DEFAULT_PROVIDER=custom
CUSTOM_API_KEY=<YOUR_API_KEY>
CUSTOM_BASE_URL=https://api.together.xyz/v1
CUSTOM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
LFL_CUSTOM_LABEL=Together
```

The base URL should end where OpenAI's does, at `/v1`.

### Ollama

No key and no account. Install from [ollama.com](https://ollama.com/download), then:

```bash
ollama pull qwen2.5:0.5b-instruct
curl http://localhost:11434/api/tags
```

After a real deploy, the default Ollama model switches to your fine-tune automatically. Leave Ollama on localhost: it has no authentication, so binding it to `0.0.0.0` exposes it to your whole network.

### Mock

`mock` needs nothing. It returns canned replies, and the tests and dry-run evaluation use it.

---

## How keys are resolved

![How settings and keys are resolved: the shell and .env feed Settings; providers, the pull stage and /api/health read it; keys go to provider APIs and Hugging Face, and the health endpoint masks the token for the System panel](images/config-secrets.svg)

Each credential can be set two ways. The project's own `LFL_` name wins; the vendor's standard name is the fallback. A value in the shell beats the same name in `.env`.

| Setting | Project name | Standard fallback |
|---|---|---|
| DeepSeek key | `LFL_DEEPSEEK_API_KEY` | `DEEPSEEK_API_KEY` |
| Anthropic key | `LFL_ANTHROPIC_API_KEY` | `ANTHROPIC_API_KEY` |
| Anthropic URL | `LFL_ANTHROPIC_BASE_URL` | `ANTHROPIC_BASE_URL` |
| OpenAI key | `LFL_OPENAI_API_KEY` | `OPENAI_API_KEY` |
| Custom key | `LFL_CUSTOM_API_KEY` | `CUSTOM_API_KEY` |
| Hugging Face token | `LFL_HF_TOKEN` | `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` |

The same pattern covers every `*_BASE_URL` and `*_MODEL`. If you already export the standard names for the official SDKs, there is nothing to configure. Only these names are read: settings written for other tools (for example `PROVIDERS__DEEPSEEK__MODEL`) are ignored.

---

## Checking it works

```bash
curl http://localhost:8000/api/providers
```

Each entry shows `configured: true/false` and the model it will use. The **System** panel in the console shows the same, with keys masked. A misconfigured provider still answers in the chat panel, with a sentence saying what is missing.

> [!TIP]
> **A setting in your shell beats `.env`.** If `ANTHROPIC_BASE_URL` (or any key) is also set as an environment variable, for example by another tool, that value wins. Error messages name the server that answered (`... at api.anthropic.com returned HTTP 401`). If that is not the server in your `.env`, clear the variable in that terminal with `Remove-Item Env:ANTHROPIC_BASE_URL` (PowerShell) or `unset ANTHROPIC_BASE_URL`, then restart the backend.

---

## Adding a provider

For an OpenAI-compatible service, a subclass is enough:

```python
# providers/openai_compatible.py
class GroqProvider(OpenAICompatible):
    name = "groq"
    service = "the Groq API"
    creds_attr = "groq"
    note = "Llama and Mixtral on Groq. Needs GROQ_API_KEY."
```

Then:

1. Add a `groq()` resolver to `Settings` in `core/config.py` that returns `(key, base_url, model)`, like `deepseek()`.
2. Add `"groq": GroqProvider` to `_FACTORIES` in `providers/registry.py`.
3. Add the new env names to `CREDENTIAL_VARS` in `tests/conftest.py`, and the new name to the provider-set assertions in `tests/test_api.py` and `tests/test_providers.py`.

---

<sub>Maintained by Asad Aslam.</sub>
