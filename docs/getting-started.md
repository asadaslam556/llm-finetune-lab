# Getting started

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Node](https://img.shields.io/badge/Node-20.19%2B-339933?logo=nodedotjs&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-for%20real%20runs-76B900?logo=nvidia&logoColor=white)
![Colab](https://img.shields.io/badge/Colab-free%20T4-F9AB00?logo=googlecolab&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Hub-FFD21E?logo=huggingface&logoColor=black)
![llama.cpp](https://img.shields.io/badge/llama.cpp-GGUF-000000)
![Ollama](https://img.shields.io/badge/Ollama-local-000000?logo=ollama&logoColor=white)

From a fresh clone to a fine-tuned model you can chat with.

## Contents

- [Pick your path](#pick-your-path)
- [1. Install](#1-install)
- [2. First dry run](#2-first-dry-run)
- [3. Chat with a model](#3-chat-with-a-model)
- [4. Real run on your own GPU](#4-real-run-on-your-own-gpu)
- [5. Real run on Colab](#5-real-run-on-colab)
- [6. Export and deploy](#6-export-and-deploy)
- [Hugging Face tokens](#hugging-face-tokens)
- [Configuration reference](#configuration-reference)
- [Stopping safely](#stopping-safely)
- [Common mistakes](#common-mistakes)
- [Troubleshooting](#troubleshooting)

---

## Pick your path

![From clone to chat: install, dry run, then a real run on your own NVIDIA GPU or on Colab, then export and deploy to Ollama and chat with nimbus-support](images/getting-started.svg)

Start with a dry run on any machine. Every stage runs with no GPU and no downloads, and simulated numbers are labelled as simulated. Then train for real on your own NVIDIA GPU, or on a free Colab T4 if you have none. Either way the export runs on your own machine, next to Ollama.

---

## 1. Install

You need **Python 3.11+**, **Node 20.19+** (or 22.12+, which Vite 8 requires) and **git**.

```bash
git clone https://github.com/asadaslam556/llm-finetune-lab.git
cd llm-finetune-lab
python -m venv .venv
```

Activate the virtual environment:

| Shell | Command |
|---|---|
| macOS / Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |

Then install the app:

```bash
pip install -e ".[dev]"
npm install --prefix web
cp .env.example .env   # first time only: this overwrites an existing .env
```

This is the light install: API, console, and the whole pipeline in dry mode. No torch.

> [!TIP]
> **Windows.** If PowerShell says running scripts is disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. `cp` works in PowerShell; in `cmd.exe` use `copy .env.example .env`. If `python` opens the Microsoft Store, use `py` instead.

---

## 2. First dry run

Start the API:

```bash
uvicorn finetune_lab.api.app:app --reload --port 8000
```

In a second terminal, the console:

```bash
npm run dev --prefix web
```

Open **http://localhost:5173**, click **Start dry run**, and watch the seven stages go green. The [demo in the README](../README.md#demo) shows the whole walkthrough. Click any stage for its metrics. The **Profile & plan** stage tells you whether this machine could run real QLoRA, and why not if it cannot.

Prefer the terminal?

```bash
finetune-lab run
```

---

## 3. Chat with a model

The chat panel works before you train anything. Set a provider in `.env` and restart the API:

```env
LFL_DEFAULT_PROVIDER=deepseek
DEEPSEEK_API_KEY=<YOUR_API_KEY>
```

Or run Ollama locally with no key at all:

```bash
ollama pull qwen2.5:0.5b-instruct
```

Every provider, and how keys are resolved, is in [providers.md](providers.md).

---

## 4. Real run on your own GPU

Install the training stack, including bitsandbytes for 4-bit:

```bash
pip install -e ".[train,quant]"
```

Check that torch sees the GPU, then check the plan before committing twenty minutes to it:

```bash
python -c "import torch; print(torch.cuda.is_available())"
finetune-lab plan
```

You want `True`, then `"strategy": "qlora"` and `"downgraded": false`. If the plan downgraded, the `reason` field says what is missing. Then:

```bash
finetune-lab run --real
```

or click **Start real run** in the console. bitsandbytes 0.43 and newer ship Windows wheels, so `pip install bitsandbytes` works there too.

---

## 5. Real run on Colab

No NVIDIA GPU? Open [`notebooks/train_on_colab.ipynb`](../notebooks/train_on_colab.ipynb) in Colab, switch the runtime to a **T4 GPU**, and run the cells. It clones this repo, trains **Qwen2.5-1.5B-Instruct** (Apache-2.0) with the same pipeline code (stages 1 to 6), and saves the adapter as `nimbus-adapter.zip` (about 70 MB). Download it from the file browser before you close the tab; Colab wipes its disk when the session ends.

Back on your machine, set the **same base model** in `.env`, or the merge fails:

```env
LFL_BASE_MODEL_HF=Qwen/Qwen2.5-1.5B-Instruct
LFL_OLLAMA_BASE_TAG=qwen2.5:1.5b-instruct
```

Then create a run directory with a dry run, unzip the adapter into it, and export for real:

```bash
pip install -e ".[train]"
finetune-lab run
# unzip nimbus-adapter.zip into the newest artifacts/<run_id>/adapter/
finetune-lab run --real --stages export_deploy
```

---

## 6. Export and deploy

The export stage merges the adapter into a full-precision base, converts it to GGUF and registers it with Ollama.

![Where each part runs: GitHub and Colab train the adapter, the PC merges it, llama.cpp converts it to GGUF, and Ollama serves nimbus-support to the FastAPI app](images/deployment.svg)

It needs two external tools:

1. **llama.cpp**, for the GGUF conversion. Install only the converter's requirements; the full `requirements.txt` pins older torch and transformers:

   ```bash
   git clone https://github.com/ggml-org/llama.cpp
   pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
   ```

   The stage finds `convert_hf_to_gguf.py` in `~/llama.cpp`, `./llama.cpp` or on your `PATH`. Anywhere else, point to it in `.env`:

   ```env
   LFL_GGUF_CONVERT_SCRIPT=/path/to/llama.cpp/convert_hf_to_gguf.py
   ```

2. **Ollama**, from [ollama.com/download](https://ollama.com/download).

When the stage finishes, it writes `artifacts/deployed.txt`, and the Ollama provider switches to your model automatically:

```bash
ollama run nimbus-support
```

If a tool is missing, the stage fails with instructions and leaves `DEPLOY_PLAN.md` with the exact commands to finish by hand.

> [!NOTE]
> Merging needs the base model in full precision in RAM: about 3 GB for 1.5B, about 15 GB for 7B. That is why the Colab notebook defaults to 1.5B.

---

## Hugging Face tokens

The default Qwen models are public, so you need **no token** for the project as it ships.

| Situation | Token? |
|---|---|
| Public models: Qwen, Mistral, most of the Hub | No |
| Gated models: Llama, Gemma | Yes, after accepting the licence on the model page |
| Private repos, including your own | Yes |

To get one, sign up at [huggingface.co](https://huggingface.co/join) and verify your email (an unverified email gives 401s that look like a bad token). Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Then either log in once:

```bash
hf auth login
```

or put it in `.env`:

```env
HF_TOKEN=<YOUR_HF_TOKEN>
```

On Colab, add a secret named `HF_TOKEN` with the key icon in the left sidebar; the notebook reads it from there. If a download fails, the pull stage says whether the token is missing or the licence is not accepted.

---

## Configuration reference

Every setting is an env var with the `LFL_` prefix, read from the shell first and then from `.env` (in the working directory or the repo root). Credentials also accept the vendor's standard name as a fallback, listed in [providers.md](providers.md#how-keys-are-resolved). `.env.example` lists the names; `.env` is gitignored.

**Server**

| Variable | Default | Purpose |
|---|---|---|
| `LFL_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Host names the API answers; blocks DNS rebinding |
| `LFL_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Browser origins allowed to call the API |
| `LFL_REQUEST_TIMEOUT_S` | `60` | Timeout for each provider HTTP call |
| `LFL_MAX_OUTPUT_TOKENS` | `1024` | Reply length cap sent to hosted providers |
| `LFL_API_PORT` | `8000` | Read by the Vite dev server only: where it proxies `/api` |

**Providers**

| Variable | Default | Purpose |
|---|---|---|
| `LFL_DEFAULT_PROVIDER` | `ollama` | `ollama`, `deepseek`, `anthropic`, `openai`, `custom` or `mock` |
| `LFL_OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `LFL_OLLAMA_BIN` | `ollama` | Ollama executable used by the export stage |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` | empty, `https://api.deepseek.com/v1`, `deepseek-chat` | DeepSeek |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL` | empty, `https://api.anthropic.com`, `claude-sonnet-5` | Anthropic or a gateway |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | empty, `https://api.openai.com/v1`, `gpt-4o-mini` | OpenAI |
| `CUSTOM_API_KEY`, `CUSTOM_BASE_URL`, `CUSTOM_MODEL` | empty | Any OpenAI-compatible endpoint |
| `LFL_CUSTOM_LABEL` | `custom OpenAI-compatible endpoint` | Name shown for the custom provider |
| `HF_TOKEN` | empty | Gated or private Hugging Face repos |

**Models and paths**

| Variable | Default | Purpose |
|---|---|---|
| `LFL_BASE_MODEL_HF` | `Qwen/Qwen2.5-0.5B-Instruct` | Base model to fine-tune |
| `LFL_OLLAMA_BASE_TAG` | `qwen2.5:0.5b-instruct` | Same base in Ollama, used before you deploy |
| `LFL_DEPLOY_NAME` | `nimbus-support` | Name of the fine-tune in Ollama |
| `LFL_DATA_DIR` | `data` | Seed data folder |
| `LFL_ARTIFACTS_DIR` | `artifacts` | Run output folder |

**Quantization and training**

| Variable | Default | Purpose |
|---|---|---|
| `LFL_FINETUNE_STRATEGY` | `qlora` | `qlora` or `lora` |
| `LFL_QUANT_BITS` | `4` | `4` or `8` |
| `LFL_QUANT_TYPE` | `nf4` | `nf4` or `fp4` |
| `LFL_DOUBLE_QUANT` | `true` | Quantize the quantization constants too |
| `LFL_COMPUTE_DTYPE` | `auto` | `auto` (bf16 on Ampere and newer, else fp16), `bfloat16`, `float16` |
| `LFL_GRADIENT_CHECKPOINTING` | `true` | Trade compute for memory |
| `LFL_LORA_R`, `LFL_LORA_ALPHA`, `LFL_LORA_DROPOUT` | `16`, `32`, `0.05` | Adapter rank, scale, dropout |
| `LFL_LORA_TARGET_MODULES` | all 7 linear layers | Attention and MLP projections |
| `LFL_LEARNING_RATE` | `0.0002` | |
| `LFL_NUM_EPOCHS` | `1` | Passes over the training data |
| `LFL_BATCH_SIZE` | `4` | Effective batch, split by the profile stage |
| `LFL_MAX_SEQ_LEN` | `512` | Tokens per example |
| `LFL_SEED` | `42` | Split and training seed |
| `LFL_WARMUP_RATIO` | `0.03` | Warmup share of total steps |
| `LFL_LR_SCHEDULER` | `cosine` | |
| `LFL_OPTIMIZER` | `paged_adamw_8bit` | Becomes `adamw_torch` without bitsandbytes |
| `LFL_VAL_SPLIT` | `0.1` | Share of rows held out for evaluation |
| `LFL_MIN_ANSWER_CHARS` | `20` | Prepare drops shorter answers |

**Export**

| Variable | Default | Purpose |
|---|---|---|
| `LFL_GGUF_CONVERT_SCRIPT` | empty (searched for) | Path to `convert_hf_to_gguf.py` |
| `LFL_GGUF_OUTTYPE` | `q8_0` | GGUF quantization type |

---

## Stopping safely

- Press **Ctrl + C** in each terminal and wait for the prompt to come back.
- Stopping the backend in the middle of a run is safe. The status file is written atomically, and the next run starts cleanly. The interrupted run stays on disk under `artifacts/`.

## Common mistakes

| Mistake | What happens | Do this instead |
|---|---|---|
| Running `cp .env.example .env` when `.env` exists | Your real keys are overwritten | Run it only the first time |
| Starting both servers in one terminal | The first one blocks the second | One terminal each |
| Opening http://localhost:8000 | You see the API docs, not the app | Open http://localhost:5173 |
| Editing `.env` without restarting the backend | Nothing changes | Ctrl + C, then start uvicorn again |
| Forgetting to activate the venv | `No module named uvicorn` | Activate `.venv` first |
| A different base model in `.env` than on Colab | The merge in the export stage fails | Use the same `LFL_BASE_MODEL_HF` in both places |
| Pasting a key into an issue, chat or screenshot | The key is exposed | Revoke it at the provider immediately |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Console says "backend offline" | Start uvicorn on port 8000 |
| `WinError 10013` or `address already in use` | Another program owns port 8000 (often a Docker container: check `docker ps`). Start uvicorn with `--port 8001` and the console with `LFL_API_PORT=8001` (PowerShell: `$env:LFL_API_PORT="8001"`) |
| Profile says "downgraded from QLoRA" | Read the reason. Usually no CUDA torch, or `pip install bitsandbytes` |
| `CUDA out of memory` | `LFL_BATCH_SIZE=1` or `LFL_MAX_SEQ_LEN=256`. On LoRA, switching to QLoRA helps most |
| Pull stage 401 or 403 | Gated model. Accept the licence on the model page and set `HF_TOKEN` |
| Chat says "no key" | Put the key in `.env` and restart the API |
| Chat says HTTP 401 from an unexpected server | A shell variable overrides `.env`. See [providers.md](providers.md#checking-it-works) |
| Chat says the model "used the whole reply budget" | A reasoning model (for example `deepseek-flash`) spent `LFL_MAX_OUTPUT_TOKENS` on hidden thinking. Use `DEEPSEEK_MODEL=deepseek-chat`, or set `LFL_MAX_OUTPUT_TOKENS=4096` |
| Chat says "could not reach" a gateway | Check the base URL, and whether you need a VPN or proxy |
| Export cannot find the converter | Set `LFL_GGUF_CONVERT_SCRIPT` to the full file path |
| Chat says Ollama "does not know the model" | Another Ollama answers on port 11434, often one inside Docker. Stop it, or run `ollama create` again against the right one |

`GET http://localhost:8000/api/health` runs every check at once; the **System** panel in the console shows the same.

---

<sub>Maintained by Asad Aslam.</sub>
