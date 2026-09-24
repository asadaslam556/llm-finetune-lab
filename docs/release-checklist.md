# Release checklist

![Gitleaks](https://img.shields.io/badge/Gitleaks-secret%20scan-CC3333)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-D7FF64?logo=ruff&logoColor=black)

Run this top to bottom before any push, tag or release. Every step says what a pass looks like. If a step fails, stop and fix it before going on.

Commands are for **Windows PowerShell**. On macOS or Linux the only differences are noted inline.

---

## 1. Environment

```powershell
py --version        # macOS/Linux: python3 --version
node --version
git --version
```

**Pass:** Python 3.11 or newer, Node 20.19+ or 22.12+, and any recent git.

## 2. Backend install

```powershell
py -3.12 -m venv .venv                 # first time only
.venv\Scripts\Activate.ps1             # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

**Pass:** the prompt starts with `(.venv)` and pip ends with `Successfully installed`.

## 3. Frontend install

```powershell
npm ci --prefix web
```

**Pass:** it ends with `found 0 vulnerabilities`. `npm ci` installs exactly what `web/package-lock.json` says.

Stop any running `npm run dev` first (Ctrl + C). On Windows, a running dev server locks files in `web/node_modules`, and `npm ci` then fails with `EPERM: operation not permitted`.

## 4. Tests

```powershell
pytest
```

**Pass:** `N passed`, with no `failed` or `error`. Takes about two minutes. No GPU needed.

## 5. Lint and format

```powershell
ruff check src tests
ruff format --check src tests
```

**Pass:** `All checks passed!` and `N files already formatted`. To fix formatting, run `ruff format src tests`.

## 6. Build

```powershell
npm run build --prefix web
```

**Pass:** it ends with `built in ...`.

## 7. Secret scan (Gitleaks)

Install once:

```powershell
winget install --id Gitleaks.Gitleaks -e     # macOS: brew install gitleaks
```

Close and reopen the terminal, then check with `gitleaks version`.

**Scan everything you are about to publish.** Run this after `git add .` and before `git commit`:

```powershell
gitleaks git --pre-commit --staged --redact -v
```

**Scan the git history.** Run this after committing and before pushing:

```powershell
gitleaks git --redact -v
```

**Pass:** both print `no leaks found`.

- The test file has one fake token marked `gitleaks:allow`. Gitleaks skips it on purpose.
- If anything else is found, do **not** push. Remove the secret, and if it was ever real, revoke it at the provider first.

Optional whole-folder scan:

```powershell
gitleaks dir --redact -v .
```

This one also reads files git ignores, such as `.env`, `.venv` and `web/node_modules`. Findings in those paths are expected and are never published. Findings anywhere else are a problem.

## 8. Local run

Two terminals, both in the project folder.

Terminal 1 (venv active):

```powershell
uvicorn finetune_lab.api.app:app --port 8000
```

Terminal 2:

```powershell
npm run dev --prefix web
```

**Pass:** terminal 1 prints `Application startup complete`, and terminal 2 prints `Local: http://localhost:5173/`.

## 9. Quick manual check

Open **http://localhost:5173**:

1. Click **Start dry run**. All seven stages turn green, and the header chip says `done`.
2. Click **Prepare & format**. It shows `train_rows` and `val_rows`.
3. In **Chat console**, pick `mock` and send "How do I pin a note?". A reply appears.
4. The **System** panel shows `ok` or `degraded`. Degraded is fine; it usually just means Ollama is not running.
5. Press **Ctrl + C** in both terminals.

## 10. Git status review

```powershell
git status
git diff --cached --stat
```

**Pass:** none of these appear in the list:

| Must never be committed | Why |
|---|---|
| `.env` (any `.env.*` except `.env.example`) | Real API keys |
| `.venv/`, `web/node_modules/`, `web/dist/` | Installed or built files |
| `artifacts/`, `*.gguf`, `*.safetensors`, `nimbus-adapter.zip` | Model files and run output |
| `.claude/`, `CLAUDE.md`, `CLAUDE.local.md` | Local AI-assistant config and notes |

`git check-ignore -v .env` should print the `.gitignore` line that hides it.

---

## Still not verified by CI

- The real training path on a GPU (see the verification status note in the [README](../README.md#what-it-does)).
- Keyboard-only and screen-reader use of the console.
- Layout on a real phone or tablet.
