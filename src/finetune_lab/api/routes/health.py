"""Health checks that actually check things.

Two endpoints, because they answer different questions:

  /api/health/live   is this process alive? Touches no dependencies, so it
                     cannot fail while the server can still answer. Point a
                     process supervisor at this one.
  /api/health        is this process able to do its job? Probes every
                     dependency for real: writes a file to the artifacts dir,
                     opens a socket to Ollama, reads the dataset, parses the
                     status file, checks provider credentials.

Rules for /api/health:
  * each check is isolated and timed. One blowing up fails that check, it
    never takes down the endpoint
  * a required dep down means status "error" and HTTP 503
  * an optional dep down means "degraded" and HTTP 200, because you can still
    work. No Ollama just means you chat with a cloud provider or the mock
  * network probes get their own short timeout, so a hung Ollama cannot hang
    the health endpoint along with it
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Response

from ... import __version__
from ...core.config import get_settings
from ...core.logging import log, mask
from ...pipeline.runner import get_store, is_running
from ...providers.registry import list_providers

router = APIRouter(tags=["health"])

APP_NAME = "llm-finetune-lab"
PROBE_TIMEOUT_S = 2.0  # a health check nobody wants to wait for is one nobody runs

TRAINING_MODULES = ("torch", "transformers", "peft", "datasets", "huggingface_hub")


@dataclass
class Check:
    name: str
    ok: bool
    required: bool
    detail: str
    ms: float = 0.0


def _timed(name: str, required: bool, fn) -> Check:
    """Run one probe, catch everything, record how long it took."""
    t0 = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception as e:  # a broken probe is a failed check, not a 500
        log.warning("health check %s raised: %s", name, e)
        ok, detail = False, f"{type(e).__name__}: {e}"
    return Check(name, ok, required, detail, round((time.perf_counter() - t0) * 1000, 1))


def _last_profile() -> dict | None:
    """Most recent profile report, if there is one. Never raises."""
    run_dir = get_store().read().get("run_dir")
    if not run_dir:
        return None
    try:
        with open(Path(run_dir) / "profile.json", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


# --- individual probes -------------------------------------------------------


def _check_artifacts_writable() -> tuple[bool, str]:
    """Actually write a file. A directory existing proves nothing about permissions."""
    d = get_settings().artifacts_dir
    d.mkdir(parents=True, exist_ok=True)
    probe = d / ".health_probe"
    probe.write_text(str(os.getpid()), encoding="utf-8")
    probe.unlink()
    return True, f"writable at {d}"


def _check_dataset() -> tuple[bool, str]:
    raw = get_settings().data_dir / "raw"
    if not raw.exists():
        return False, f"{raw} does not exist. Run scripts/make_seed_data.py."
    files = sorted(raw.glob("*.jsonl"))
    if not files:
        return False, f"no .jsonl files in {raw}. Run scripts/make_seed_data.py."
    with open(files[0], encoding="utf-8-sig") as f:
        lines = sum(1 for _ in f)
    return True, f"{len(files)} file(s), {lines} rows in {files[0].name}"


def _check_status_file() -> tuple[bool, str]:
    return True, f"readable, state={get_store().read().get('state')}"


def _check_ollama() -> tuple[bool, str]:
    """Open a real socket, then ask Ollama what it has pulled.

    Socket first, because "nothing is listening" and "something is listening
    but it is not Ollama" are different problems with different fixes.
    """
    s = get_settings()
    parsed = urlparse(s.ollama_host)
    host, port = parsed.hostname or "localhost", parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=PROBE_TIMEOUT_S):
            pass
    except OSError as e:
        return (
            False,
            f"nothing listening on {host}:{port} ({type(e).__name__}). Start `ollama serve`.",
        )

    r = httpx.get(f"{s.ollama_host.rstrip('/')}/api/tags", timeout=PROBE_TIMEOUT_S)
    if r.status_code != 200:
        return False, f"{host}:{port} answered HTTP {r.status_code}. Is that really Ollama?"
    tags = [m.get("name", "") for m in r.json().get("models", [])]
    marker = s.artifacts_dir / "deployed.txt"
    target = marker.read_text(encoding="utf-8").strip() if marker.exists() else s.ollama_base_tag
    have = any(t == target or t.split(":")[0] == target.split(":")[0] for t in tags)
    if not have:
        return False, (
            f"up, but '{target}' is not pulled ({len(tags)} other model(s)). "
            f"Try: ollama pull {s.ollama_base_tag}"
        )
    return True, f"up, {len(tags)} model(s), '{target}' present"


def _check_providers() -> tuple[bool, str]:
    """Config level only. No network here, that is the ollama check's job."""
    s = get_settings()
    infos = {p["name"]: p for p in list_providers()}
    if s.default_provider not in infos:
        return False, f"LFL_DEFAULT_PROVIDER='{s.default_provider}' is not a known provider"
    ready = [n for n, p in infos.items() if p["configured"]]
    if not infos[s.default_provider]["configured"]:
        return False, (
            f"default provider '{s.default_provider}' has no credentials; "
            f"ready: {', '.join(ready) or 'none'}"
        )
    return True, f"default '{s.default_provider}' configured; ready: {', '.join(ready)}"


def _check_hf_auth() -> tuple[bool, str]:
    """Optional. Public models need no token at all, so a missing one is
    normal. This exists so that when a gated model does fail, you can see at
    a glance whether auth was ever configured."""
    token = get_settings().resolved_hf_token()
    if not token:
        return True, "no token set (fine for public models like the Qwen default)"
    if not token.startswith("hf_"):
        return False, "token does not start with 'hf_'. Is that really an HF token?"
    return True, f"token set ({mask(token)}), needed only for gated or private repos"


def _check_training_extras() -> tuple[bool, str]:
    """Optional by design. Dry runs need none of this installed.

    Deliberately uses find_spec rather than importing anything. Importing
    torch here would pull about a gigabyte into the API process and cost a
    second and a half on the first call, which is a rotten thing to do to an
    endpoint the UI polls. For hardware detail, read the last run's profile
    report; the profile stage probes properly.
    """
    missing = [m for m in TRAINING_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        return False, (
            f"not installed: {', '.join(missing)}. Dry runs work; "
            "real runs need pip install -e '.[train]'."
        )
    profile = _last_profile()
    hw = (profile or {}).get("hardware", {})
    if hw.get("gpu"):
        return True, f"installed, last run saw {hw['gpu']} ({hw.get('vram_gb')} GB)"
    if profile:
        return True, "installed, last run found no GPU (real training will be slow)"
    return True, "installed (run the profile stage to see what hardware it finds)"


def _check_quantization() -> tuple[bool, str]:
    """Can this machine actually do QLoRA?

    Optional, and this is the check people will read most often after
    switching strategies. Same find_spec discipline as above: bitsandbytes
    imports CUDA libraries, which is not something a polled endpoint should do.
    """
    s = get_settings()
    if s.finetune_strategy == "lora":
        return True, "strategy is LoRA by config, so 4-bit is not needed"

    if importlib.util.find_spec("bitsandbytes") is None:
        return False, (
            "bitsandbytes is not installed, so real runs fall back to LoRA. "
            "Install it with: pip install bitsandbytes"
        )
    profile = _last_profile()
    plan = (profile or {}).get("plan")
    if not plan:
        return True, "bitsandbytes installed (run the profile stage to confirm the GPU can use it)"
    if plan.get("downgraded"):
        return False, f"last run downgraded to LoRA: {plan.get('reason')}"
    return True, (
        f"{plan.get('bits')}-bit {plan.get('quant_type')}, compute in {plan.get('compute_dtype')}"
    )


# --- endpoints ---------------------------------------------------------------


@router.get("/api/health/live")
def liveness():
    """Pure liveness. Touches nothing, so it cannot lie about being up."""
    return {"status": "ok", "app": APP_NAME, "version": __version__}


@router.get("/api/health")
def health(response: Response):
    s = get_settings()
    checks = [
        _timed("storage.artifacts", True, _check_artifacts_writable),
        _timed("storage.dataset", False, _check_dataset),
        _timed("storage.status_file", True, _check_status_file),
        _timed("providers.config", True, _check_providers),
        _timed("providers.ollama", False, _check_ollama),
        _timed("providers.huggingface", False, _check_hf_auth),
        _timed("training.extras", False, _check_training_extras),
        _timed("training.quantization", False, _check_quantization),
    ]

    required_down = [c.name for c in checks if c.required and not c.ok]
    optional_down = [c.name for c in checks if not c.required and not c.ok]

    if required_down:
        status = "error"
        response.status_code = 503
    elif optional_down:
        status = "degraded"
    else:
        status = "ok"

    return {
        "status": status,
        "app": APP_NAME,
        "version": __version__,
        "failing": required_down + optional_down,
        "pipeline_busy": is_running(),
        "config": {
            "default_provider": s.default_provider,
            "base_model": s.base_model_hf,
            "deploy_name": s.deploy_name,
            "ollama_host": s.ollama_host,
            "finetune_strategy": s.finetune_strategy,
            "quantization": f"{s.quant_bits}-bit {s.quant_type}"
            if s.finetune_strategy == "qlora"
            else "none",
        },
        "checks": [asdict(c) for c in checks],
    }
