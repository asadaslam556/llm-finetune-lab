# Security

![Local only](https://img.shields.io/badge/API-127.0.0.1%20only-555555)
![FastAPI](https://img.shields.io/badge/FastAPI-TrustedHost-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-request%20limits-E92063?logo=pydantic&logoColor=white)
![Gitleaks](https://img.shields.io/badge/Gitleaks-secret%20scan-CC3333)

## What this project is

llm-finetune-lab is a local developer tool. The API has **no login** and is meant to run on your own machine, bound to `127.0.0.1`. Do not expose it to a network or the internet. If you need to, put it behind a reverse proxy that adds authentication.

Built-in local protections:

- The API answers only requests addressed to the host names in `LFL_ALLOWED_HOSTS` (default `localhost` and `127.0.0.1`). This blocks DNS-rebinding attacks from a browser tab.
- Chat requests are size-limited (100 messages, 20,000 characters each), because they are forwarded to paid APIs. Caller-supplied request IDs are validated.
- CI runs with read-only permissions, and its GitHub Actions are pinned to exact commits.

## Keeping your keys safe

![How settings and keys are resolved: the shell and .env feed Settings; providers, the pull stage and /api/health read it; keys go to provider APIs and Hugging Face, and the health endpoint masks the token for the System panel](docs/images/config-secrets.svg)

- Keys are read from `.env` or the shell environment and are never logged or returned in full; the health endpoint masks them.
- `.env` is gitignored and `.env.example` holds names only. Never put keys in code, issues or screenshots.
- Run a secret scan before every push, as described in [CONTRIBUTING.md](CONTRIBUTING.md#secret-scan).
- Use one key per place it is used (laptop, Colab), so revoking one does not break the others.

**If a key leaks**, revoke it at the provider first. That is the step that matters: deleting it from git does not make it safe again, and public repos are scraped within minutes. Then issue a replacement and update `.env`. This applies to keys pasted into chats, tickets and screenshots too.

## Reporting a vulnerability

Please use GitHub's private reporting: **Security → Report a vulnerability** on this repository. Do not open a public issue for security problems, and never include real keys in a report.

This is a personal open-source project maintained on a best-effort basis. There is no guaranteed response time.
