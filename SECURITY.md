# Security

## What this project is

llm-finetune-lab is a local developer tool. The API has **no login** and is meant to run on your own machine, bound to `127.0.0.1`. Do not expose it to a network or the internet. If you need to, put it behind a reverse proxy that adds authentication.

Built-in local protections:

- The API answers only requests addressed to the host names in `LFL_ALLOWED_HOSTS` (default `localhost` and `127.0.0.1`). This blocks DNS-rebinding attacks from a browser tab.
- Chat requests are size-limited, because they are forwarded to paid APIs.
- API keys are read from `.env` or the environment and are never logged in full.

## Keeping your keys safe

- Put keys in `.env`, which git ignores. Never put them in code, issues or screenshots.
- Run a secret scan before every push. See [docs/release-checklist.md](docs/release-checklist.md).
- If a key leaks, **revoke it at the provider first**. Deleting it from git does not make it safe again.

## Reporting a vulnerability

Please use GitHub's private reporting: **Security → Report a vulnerability** on this repository. Do not open a public issue for security problems, and never include real keys in a report.

This is a personal open-source project maintained on a best-effort basis. There is no guaranteed response time.
