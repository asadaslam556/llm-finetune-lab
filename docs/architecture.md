# Architecture

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-499848)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)
![HTTPX](https://img.shields.io/badge/HTTPX-client-2B5B84)

How the pieces fit, and why they are shaped the way they are. Every diagram here is drawn from the code it names.

---

## Contents

- [System overview](#system-overview)
- [Layers](#layers)
- [A run, end to end](#a-run-end-to-end)
- [Run and stage status](#run-and-stage-status)
- [The run directory](#the-run-directory)
- [Chat and Compare mode](#chat-and-compare-mode)
- [Health checks](#health-checks)
- [Configuration and secrets](#configuration-and-secrets)
- [HTTP API reference](#http-api-reference)
- [Frontend](#frontend)
- [Design decisions](#design-decisions)
- [Extending it](#extending-it)
- [Known limitations](#known-limitations)

---

## System overview

![System overview: the React console and the CLI both drive the runner; the FastAPI app reaches chat providers through a registry; stages write the run directory, download from Hugging Face and deploy to Ollama](images/system-overview.svg)

- **Entry points.** The browser console and the `finetune-lab` CLI drive the same runner. The API only validates input and shapes responses.
- **Pipeline.** One run at a time, in a background thread. Status is one JSON file, written atomically.
- **Dashed edges are real runs only.** Dry runs never download from Hugging Face or deploy to Ollama.

---

## Layers

![Package layers: web calls api over HTTP; api and cli call pipeline; api and pipeline call providers; pipeline and providers use core](images/layers.svg)

Imports only point downward. `core` imports nothing else from the project, so every package can use it (the API too; that edge is left out of the drawing to keep it readable). `pipeline` never imports `api`, which is why the CLI can drive a run with no web server. The one dashed edge is narrow on purpose: `s6_evaluate.py` uses `MockProvider` for dry-run answers.

| Package | Owns | Must not |
|---|---|---|
| `core` | Settings, logger, error types, secret masking | Import anything else from the project |
| `providers` | One `chat()` interface over every backend | Know about the pipeline |
| `pipeline` | Stages, runner, status file, QLoRA decision | Know about HTTP |
| `api` | Validation, status codes, response shapes | Hold business logic |
| `cli.py` | Terminal entry point | Duplicate pipeline logic |

---

## A run, end to end

![Starting a pipeline run: the console posts to /api/pipeline/run, the runner validates and starts a worker thread, the API returns 202, the worker runs each stage and writes status.json, and the console polls /api/pipeline/status every 1.5 seconds](images/pipeline-run.svg)

1. **Checked before the thread starts.** `start_run` validates in the caller's thread, so mistakes come back at once: an unknown stage or a missing input file is a **400**, a run already going is a **409**, and a real run without the training libraries is a **400** that says what to install.
2. **One run at a time.** A lock-guarded flag marks the runner busy and is cleared in a `finally` block. A failed stage stops the run.
3. **Why `busy` comes from the lock.** The status file says `done` a moment before the flag clears, so the console disables **Start** on `busy`, not on the file's state.

---

## Run and stage status

The run and each of its stages have their own states, all stored in `artifacts/status.json` (`pipeline/status.py`).

![Run status lifecycle: idle to running on start_run, running to done on run_finished, running to failed when a stage raises, and failed back to running on the next start_run](images/run-status.svg)

![Stage status lifecycle: pending to running on stage_started, running to done on stage_finished, running to failed on stage_failed; stages outside the chosen subset are skipped](images/stage-status.svg)

| Object | States | Notes |
|---|---|---|
| Run | `idle` → `running` → `done` or `failed` | Any finished run can be followed by a new one |
| Stage | `pending` or `skipped` → `running` → `done` or `failed` | `skipped` means not requested. After a failure, the remaining stages stay `pending` |

---

## The run directory

Each stage reads files an earlier stage wrote into `artifacts/<run_id>/`. The runner checks this before starting, so a subset run with a missing input is a 400 that names the missing file.

![What each stage reads and writes: data/raw to ingested.jsonl, then train.jsonl and val.jsonl, profile.json, the adapter, the Modelfile and the evaluation report](images/run-directory.svg)

| # | Stage | Needs | Writes | Real mode also needs |
|---|---|---|---|---|
| 1 | `ingest` | `data/raw/*.jsonl` | `ingested.jsonl` | |
| 2 | `prepare` | `ingested.jsonl` | `train.jsonl`, `val.jsonl` | |
| 3 | `pull_base` | | `artifacts/models/<model>/` (real mode, shared across runs) | network; `HF_TOKEN` for gated models |
| 4 | `profile` | `train.jsonl` | `profile.json` | |
| 5 | `finetune` | `train.jsonl`, `profile.json` | `adapter/` | torch; CUDA for QLoRA |
| 6 | `evaluate` | `val.jsonl` | `eval_report.json` | `adapter/` |
| 7 | `export_deploy` | | `Modelfile`, `DEPLOY_PLAN.md` | `adapter/`, llama.cpp, Ollama |

The source of truth is `STAGE_INPUTS` and `STAGE_OUTPUTS` in `pipeline/status.py`, plus `STAGE_INPUTS_REAL` in `pipeline/runner.py`.

```
artifacts/
├── status.json                  # live state, read by the console
├── deployed.txt                 # set after a real deploy, flips the Ollama default
├── models/Qwen__Qwen2.5-1.5B-Instruct/   # base model snapshot, shared across runs
└── 20260921-120317-d03286/
    ├── ingested.jsonl
    ├── train.jsonl, val.jsonl
    ├── profile.json             # hardware + QLoRA plan + VRAM estimate
    ├── adapter/                 # LoRA weights + ADAPTER_INFO.json
    ├── eval_report.json
    ├── merged/                  # real mode only
    ├── model.gguf               # real mode only
    ├── Modelfile
    └── DEPLOY_PLAN.md
```

A run that skips `ingest` continues in the newest run directory, so re-running only `evaluate` works without redoing everything.

---

## Chat and Compare mode

![Chat in Compare mode: the console sends two POST /api/chat requests at once; the API looks up each provider in the registry, calls Ollama and Anthropic, and returns ok, reply and latency for each](images/chat-compare.svg)

- **Two requests, sent together.** Each carries only its own provider's history, so one model never reads another's replies. Answers land side by side as they arrive.
- **Outages are not errors.** A provider that is down or unconfigured returns HTTP 200 with `ok: false`, and `reply` explains what to fix.
- **Hard errors.** An unknown provider name is a 400. More than 100 messages, or a message over 20,000 characters, is a 422.

---

## Health checks

![Health check: the console calls GET /api/health every 15 seconds; the API probes the disk, the provider registry, Ollama and the Python environment, then answers 200 ok or degraded, or 503](images/health-check.svg)

`GET /api/health` runs eight timed checks. Each is isolated, network probes time out after 2 s, and nothing imports torch (`importlib.util.find_spec` only checks it is installed).

| Check | Required | What it looks at |
|---|---|---|
| `storage.artifacts` | yes | `artifacts/` is writable |
| `storage.status_file` | yes | `status.json` is readable |
| `providers.config` | yes | The default provider has its credentials |
| `storage.dataset` | no | Seed rows in `data/raw/` |
| `providers.ollama` | no | Ollama answers, and the default model (the deployed fine-tune or the base tag) is pulled |
| `providers.huggingface` | no | Token present (masked), needed only for gated repos |
| `training.extras` | no | torch, transformers, peft, datasets and huggingface_hub are installed |
| `training.quantization` | no | 4-bit QLoRA is possible on this machine |

A failed required check returns **503**; a failed optional one returns **200** with status `degraded`. `GET /api/health/live` touches nothing and always answers while the process is up.

---

## Configuration and secrets

![How settings and keys are resolved: the shell and .env feed Settings; providers, the pull stage and /api/health read it; keys go to provider APIs and Hugging Face, and the health endpoint masks the token for the System panel](images/config-secrets.svg)

For each credential, the first non-empty value wins: the `LFL_*` name in the shell, then in `.env`, then the vendor's standard name (`DEEPSEEK_API_KEY`, `HF_TOKEN`, ...) in the shell, then in `.env`, then the built-in default. Non-credential settings only read `LFL_*` names. `.env` is looked up in the working directory and the repo root.

Keys never appear in logs or API responses. The health endpoint shows a token as `hf_Abc...7654` at most, through `core.logging.mask`, and a test asserts the full token never leaks. Every setting is listed in [getting-started.md](getting-started.md#configuration-reference).

---

## HTTP API reference

Interactive docs are at **http://localhost:8000/docs** (Swagger) while the server runs.

| Method | Path | Body | Returns | Status codes |
|---|---|---|---|---|
| `GET` | `/` | | Redirect to `/docs` | 307 |
| `GET` | `/api/health/live` | | `{status, app, version}` | 200 |
| `GET` | `/api/health` | | status, failing checks, config echo, 8 timed checks | 200 ok or degraded, 503 when a required check fails |
| `GET` | `/api/providers` | | `[{name, configured, default_model, note, default}]` | 200; no network calls |
| `POST` | `/api/chat` | `{messages, provider?, model?}` | `{ok, provider, model, reply, latency_ms, error_kind}` | 200 (also for outages, with `ok: false`), 400 unknown provider, 422 too large |
| `POST` | `/api/pipeline/run` | `{dry_run: true, stages?: [...]}` | `{run_id, dry_run, stages, run_dir}` | 202 started, 400 bad stages or missing inputs, 409 busy |
| `GET` | `/api/pipeline/status` | | status file contents plus `busy` | 200; polled every 1.5 s |

Every response carries `X-Request-ID` and `X-Response-Time-Ms`. A 500 includes the request id, which matches a line in the server log. Requests pass through three middlewares, outermost first: trusted hosts (`LFL_ALLOWED_HOSTS`), CORS (`LFL_CORS_ORIGINS`), then request tracing.

Example:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"provider": "deepseek", "messages": [{"role": "user", "content": "How long do deleted notes stay in the trash?"}]}'
```

---

## Frontend

The console in `web/` is plain React with no state library. `App.jsx` polls `/api/pipeline/status` and passes the result down.

| Component | Shows |
|---|---|
| `PipelineRail` | The seven stages as nodes. The running stage ticks elapsed time from the server's start timestamp, so a reload does not reset the clock. |
| `RunControls` | **Start dry run** and **Start real run**. Disabled while the backend says busy. |
| `StageDetail` | Message and metrics for the selected or running stage |
| `ChatPanel` | Provider picker (preselected on `LFL_DEFAULT_PROVIDER`), an optional Compare provider, quick-question chips, and light formatting (line breaks, bold, headings). Each provider gets only its own history. |
| `SystemHealth` | `/api/health` results, refreshed every 15 s. Keeps the check details on a 503, which is when you need them. |

`lib/api.js` wraps `fetch` and turns FastAPI error bodies, including validation arrays, into one readable sentence. In development, Vite proxies `/api` to port 8000 (or `LFL_API_PORT`), so the frontend needs no config.

---

## Design decisions

**Stages are functions.** Each stage is `run(ctx) -> (message, metrics)`. No base class, no registry, no plugin system. A function is the easiest thing to unit-test, and the runner stays a short loop.

**Status lives in a JSON file.** The runner writes from a background thread and the API reads from request handlers. A file is the simplest shared state that also survives a server restart. Writes are atomic (temp file plus `os.replace`) and retry on Windows sharing violations.

**One run at a time.** A single GPU cannot usefully train twice at once. A second start is a 409, not a queue.

**The plan is decided once.** Stage 4 writes `profile.json`, and every later stage reads it. Nothing downstream re-probes the hardware, so what the rail shows is what actually runs. See [qlora.md](qlora.md#3-how-the-plan-is-chosen).

**Heavy imports are lazy.** `torch`, `transformers`, `peft` and `bitsandbytes` are only imported inside real-mode code paths. The API server and the whole dry run work without them.

**Failures read like sentences.** A provider that is down returns HTTP 200 with `ok: false` and an explanation in the chat bubble. A stage that fails raises `StageError` with what to do next. Tracebacks go to the server log, never the UI.

**Dry runs are honest.** Simulated numbers are labelled simulated in the rail message, the metrics, and the files on disk.

**No auth, on purpose.** The API can start training runs and has no login. It binds to `127.0.0.1` by default; keep it there, or put it behind a reverse proxy with auth. Two guards cover the local case: the trusted-host check blocks DNS-rebinding attacks from a browser tab, and chat requests are capped because they are forwarded to paid APIs.

---

## Extending it

| To add | Do this |
|---|---|
| An OpenAI-compatible provider | Subclass `OpenAICompatible` with a `creds_attr`, add a resolver to `Settings`, register it in `_FACTORIES` in `providers/registry.py`. Then add its env names to `CREDENTIAL_VARS` in `tests/conftest.py` and the new name to the provider-set assertions in `tests/test_api.py` and `tests/test_providers.py`. See [providers.md](providers.md#adding-a-provider). |
| A new stage | Write `run(ctx)`, then add it to `STAGE_ORDER`, `STAGE_LABELS`, `STAGE_INPUTS`, `STAGE_OUTPUTS` and `STAGE_FUNCS`. |
| A new metric | Add a function in `s6_evaluate.py` and put it in the report dict. |

---

## Known limitations

- The rail label for stage 5 is always "Fine-tune (QLoRA)", even when the plan fell back to LoRA. The stage message and metrics state the real strategy.
- `artifacts/deployed.txt` is one global marker. It is not tied to the base model, so after switching `LFL_BASE_MODEL_HF` the Ollama provider keeps pointing at the old fine-tune until the next deploy.
- Evaluation uses about 12 validation rows, so its scores are a sanity check, not a benchmark.
- Loss is reported when training ends, not streamed live.

---

<sub>Maintained by Asad Aslam.</sub>
