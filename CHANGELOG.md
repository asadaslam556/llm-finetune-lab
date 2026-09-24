# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## [2.0.1]

### Fixed

- Real training crashed on **transformers 5** (the version Colab installs) with `unexpected keyword argument 'warmup_ratio'`. Warmup is now passed as whole steps, which works on transformers 4 and 5.
- A Tesla T4 was treated as supporting bfloat16 because newer torch counts slow emulation. bf16 is now used only on GPUs that support it in hardware (Ampere or newer), so a T4 trains in float16.
- **Start real run** on a machine without the training libraries is now refused immediately, with a message saying what to install or to use the Colab notebook. Before, it ran two stages and then failed on an import.
- Provider errors now name the server that answered (for example `... at api.anthropic.com returned HTTP 401`). A base URL set in the shell overrides `.env`, and the error used to hide which one was in use.
- Clearer wording on the Run panel about what dry and real runs need.

### Verified

- A real run on CPU (Windows, transformers 5.17, peft 0.21, Qwen2.5-0.5B, plain-LoRA fallback) completed ingest, prepare, pull, profile, 28 training steps with falling loss, evaluation, and the adapter merge. Stopped as designed at GGUF conversion, since llama.cpp was not installed.
- **Still unverified:** 4-bit QLoRA on a GPU (Colab T4), GGUF conversion and `ollama create`.

## [2.0.0] - first public release

### Added

- Seven-stage pipeline: ingest, prepare, pull base model, profile & plan, fine-tune, evaluate, export & deploy.
- QLoRA by default (4-bit NF4 base, LoRA on every linear layer, paged 8-bit optimizer). Falls back to plain LoRA with a stated reason when the machine cannot do 4-bit.
- Training loss is computed on the assistant's reply only.
- Dry-run mode for every stage: no GPU, no downloads, simulated numbers clearly labelled.
- React ops console: live pipeline rail, stage metrics, chat panel, system health.
- `finetune-lab` CLI: `run`, `plan`, `serve`.
- Chat providers: Ollama, Anthropic, DeepSeek, OpenAI, any OpenAI-compatible endpoint, and a mock.
- Colab notebook for real training on a free T4 GPU.
- Seed dataset of 137 fictional Nimbus support tickets, with deliberate dirt for the cleaning stages.

### Security

- The API only answers `localhost` / `127.0.0.1` host names (`LFL_ALLOWED_HOSTS`), which blocks DNS rebinding.
- Chat requests are size-limited, and caller-supplied request IDs are validated.
- CI runs with read-only permissions, and its GitHub Actions are pinned to exact commits.

### Known limitations

- **The real training path is not yet verified end to end** on a GPU or Colab. The dry run is fully tested; the real-mode logic is only unit-tested without a GPU. (See 2.0.1 for what has since been verified.)
- Evaluation uses about 12 validation rows, so its scores are a sanity check, not a benchmark.
- The API has no authentication and is meant for local use only.
