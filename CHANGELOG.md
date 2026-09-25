# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## [2.1.0]

### Added

- **Compare mode** in the chat: pick a second provider and every question goes to both, answered side by side with response times. Built for putting the fine-tune next to a hosted model.
- Quick-question chips in the empty chat, and light formatting of replies (line breaks, bold, headings).
- Motion that explains state: the pipeline rail fills downward as stages finish, a progress line runs along the header, check marks draw in, replies fade in, and a typing indicator shows while a model thinks. All of it is removed under `prefers-reduced-motion`.
- README screenshots.

### Fixed

- The chat sent the whole conversation to whichever provider was selected, so one model read another's replies (and apologised for "its" earlier answer). Each provider now gets only its own turns.
- On page load the chat scrolled the whole page, hiding the header on phones. Only the chat log scrolls now.
- The phone header wrapped the title over three lines.
- The **Send** button wrapped onto its own line at common widths, and the pipeline panel stretched far below its content.
- The System panel reported "degraded" on machines without an NVIDIA GPU, telling people to install bitsandbytes, which cannot help there. It now says 4-bit is not available on this machine and points to Colab.
- The Colab notebook's download step now points to the file browser, since the automatic download can hang.

### Changed

- Drawn SVG check and cross marks replace unicode glyphs on the rail. Scrollbars, text selection and the input caret use the console palette.
- Dependabot now proposes weekly updates for Python, npm and GitHub Actions, with `react` and `react-dom` grouped so they always upgrade together.

## [2.0.1]

### Fixed

- Real training crashed on **transformers 5** (the version Colab installs) with `unexpected keyword argument 'warmup_ratio'`. Warmup is now passed as whole steps, which works on transformers 4 and 5.
- A Tesla T4 was treated as supporting bfloat16 because newer torch counts slow emulation. bf16 is now used only on GPUs that support it in hardware (Ampere or newer), so a T4 trains in float16.
- **Start real run** on a machine without the training libraries is now refused immediately, with a message saying what to install or to use the Colab notebook. Before, it ran two stages and then failed on an import.
- Provider errors now name the server that answered (for example `... at api.anthropic.com returned HTTP 401`). A base URL set in the shell overrides `.env`, and the error used to hide which one was in use.
- Clearer wording on the Run panel about what dry and real runs need.
- On Windows, reading the output of `ollama create` crashed a background thread with `UnicodeDecodeError`. External tools are now read as UTF-8.
- No more `torch_dtype is deprecated` warning on transformers 4.56 and newer.

### Verified

- **The full real path works end to end.** QLoRA training of Qwen2.5-1.5B on a free Colab T4 GPU (84 steps, final loss 1.16, overlap-F1 0.31, keyword hit rate 0.53). Then, on Windows 11 without a GPU: adapter merge, GGUF conversion with llama.cpp, and `ollama create`. The deployed model answers in the trained style and gets some facts right and others wrong.
- The real path also ran on CPU (plain-LoRA fallback, Qwen2.5-0.5B, transformers 5.17) through the merge.

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
