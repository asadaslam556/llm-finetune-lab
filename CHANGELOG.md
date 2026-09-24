# Changelog

This project follows [Semantic Versioning](https://semver.org/).

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

- **The real training path is not yet verified end to end** on a GPU or Colab. The dry run is fully tested; the real-mode logic is only unit-tested without a GPU.
- Evaluation uses about 12 validation rows, so its scores are a sanity check, not a benchmark.
- The API has no authentication and is meant for local use only.
