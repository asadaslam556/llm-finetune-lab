# Contributing

[![CI](https://img.shields.io/github/actions/workflow/status/asadaslam556/llm-finetune-lab/ci.yml?branch=main&label=CI&logo=githubactions&logoColor=white)](https://github.com/asadaslam556/llm-finetune-lab/actions/workflows/ci.yml)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-D7FF64?logo=ruff&logoColor=black)
![Gitleaks](https://img.shields.io/badge/Gitleaks-secret%20scan-CC3333)
![Dependabot](https://img.shields.io/badge/Dependabot-weekly-025E8C?logo=dependabot&logoColor=white)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-FE5196?logo=conventionalcommits&logoColor=white)](https://www.conventionalcommits.org/)

Bug reports, fixes and small improvements are welcome.

## Before you start

- For anything bigger than a small fix, open an issue first so we can agree on the approach.
- Security problems go through **Security → Report a vulnerability**, never a public issue. See [SECURITY.md](SECURITY.md).

## Set up

Follow [docs/getting-started.md](docs/getting-started.md) up to the first dry run. You need Python 3.11+ and Node 20.19+; no GPU.

## Rules for changes

![Package layers: web calls api over HTTP; api and cli call pipeline; api and pipeline call providers; pipeline and providers use core](docs/images/layers.svg)

- Keep the layering: `api` and `cli` call `pipeline` and `providers`, which call `core`. Nothing imports upward. [docs/architecture.md](docs/architecture.md#layers) explains why.
- `torch`, `transformers`, `peft`, `datasets`, `bitsandbytes` and `huggingface_hub` are imported only inside real-mode functions. CI has none of them installed, and a test checks `/api/health` works without them.
- Dry-run numbers must say they are simulated, everywhere they appear.
- Errors a person will read say what to do next. Stage failures raise `StageError`; provider outages return HTTP 200 with `ok: false`.
- Never print or log a full secret. Use `core.logging.mask`.
- Add or change seed tickets in `scripts/make_seed_data.py`, never by hand in the JSONL.
- New behaviour needs a test. Changes to training or export code also need a real run (Colab is fine), since CI cannot run it.

## Check before you push

```bash
pytest
ruff check src tests
ruff format --check src tests
npm run build --prefix web
```

`ruff format src tests` fixes formatting; `make lint` and `make fmt` are shortcuts. Stop any running `npm run dev` before `npm ci`: on Windows, the dev server locks files in `web/node_modules`.

### Secret scan

Scan what you are about to commit, then the history, with [Gitleaks](https://github.com/gitleaks/gitleaks) (`winget install Gitleaks.Gitleaks` or `brew install gitleaks`):

```bash
gitleaks git --pre-commit --staged --redact -v
gitleaks git --redact -v
```

Both should print `no leaks found`. The tests contain one fake token marked `gitleaks:allow`; it is skipped on purpose. If anything else shows up, do not push: remove it, and if it was ever a real key, revoke it at the provider first.

These must never be committed, and `.gitignore` already covers them:

| Path | Why |
|---|---|
| `.env` (any `.env.*` except `.env.example`) | Real API keys |
| `.venv/`, `web/node_modules/`, `web/dist/` | Installed or built files |
| `artifacts/`, `*.gguf`, `*.safetensors`, `nimbus-adapter.zip` | Model files and run output |

`git check-ignore -v .env` prints the `.gitignore` line that hides it.

## Continuous integration

![CI on every push and pull request: Dependabot opens weekly update PRs; each push or PR runs a backend job on Python 3.11 and 3.12 (install, ruff check, ruff format, pytest) and a frontend job on Node 22 (npm ci, Vite build)](docs/images/ci.svg)

`.github/workflows/ci.yml` runs on every push to `main` and every pull request:

| Job | Runs on | Steps |
|---|---|---|
| `backend` | Python 3.11 and 3.12 | `pip install -e ".[dev]"`, `ruff check`, `ruff format --check`, `pytest --cov` |
| `frontend` | Node 22 | `npm ci`, `npm run build` |

Both jobs run in parallel. A newer push to the same branch cancels the old run. The workflow has read-only permissions, and every action is pinned to a commit SHA. Dependabot opens weekly update pull requests for pip, npm (with `react` and `react-dom` grouped) and GitHub Actions.

## Pull requests

- One topic per pull request.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `chore:`).
- Say what you tested. For UI changes, add a screenshot.
- Never commit `.env` or any real key.
