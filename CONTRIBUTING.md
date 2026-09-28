# Contributing

Thanks for looking. Bug reports, fixes and small improvements are all welcome.

## Before you start

- For anything bigger than a small fix, open an issue first so we can agree on the approach.
- Security problems go through **Security → Report a vulnerability**, never a public issue. See [SECURITY.md](SECURITY.md).

## Set up

Follow [docs/getting-started.md](docs/getting-started.md) up to the first dry run. You need Python 3.11+ and Node 20.19+; no GPU.

## Make a change

- Keep the layering: `api` and `cli` call `pipeline` and `providers`, which call `core`. Nothing imports upward. [docs/architecture.md](docs/architecture.md) explains why.
- `torch`, `transformers`, `peft`, `datasets`, `bitsandbytes` and `huggingface_hub` are imported only inside real-mode functions. CI has none of them installed.
- Dry-run numbers must say they are simulated, everywhere they appear.
- Errors a person will read should say what to do next.
- Add or change seed tickets in `scripts/make_seed_data.py`, never by hand in the JSONL.
- New behaviour needs a test. Changes to real training or export code should also be checked with a real run (Colab is fine), since CI cannot run it.

## Check before you push

```bash
pytest
ruff check src tests
ruff format --check src tests
npm run build --prefix web
```

The full list, including a secret scan, is in [docs/release-checklist.md](docs/release-checklist.md).

## Pull requests

- One topic per pull request.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `chore:`).
- Say what you tested. For UI changes, add a screenshot.
- Never commit `.env` or any real key.
