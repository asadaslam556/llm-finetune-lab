"""Stage 3, pull the base model.

Real mode: snapshot-download the base model from Hugging Face into
artifacts/models/. huggingface_hub is imported lazily so the API server never
needs the training extras installed.

Dry mode: no network. We record what would be pulled and check whether a
local copy already exists, so the dry run still tells you something useful.

Note this pulls the full-precision weights even when you are training with
QLoRA. Quantization happens at load time in the training stage, not here, so
the same download serves both strategies and switching costs you nothing.
"""

from __future__ import annotations

from ...core.errors import StageError
from ..context import StageContext


def _download_hint(text: str, model: str, has_token: bool) -> str:
    low = text.lower()
    if any(k in low for k in ("401", "403", "gated", "authoriz")):
        if has_token:
            return (
                f" You have a token set, so it is probably the terms: open "
                f"huggingface.co/{model} and accept them, and check your token has read "
                "access to that repo."
            )
        return (
            f" This model is gated and you have no token set. Accept its terms on "
            f"huggingface.co/{model}, then run `hf auth login` or set LFL_HF_TOKEN."
        )
    if "404" in low or "repositorynotfound" in low:
        return f" Check LFL_BASE_MODEL_HF: '{model}' does not resolve to a repo."
    if any(k in low for k in ("connect", "timeout", "resolve", "network")):
        return " Looks like a network problem. Check your connection or proxy settings."
    return ""


def run(ctx: StageContext) -> tuple[str, dict]:
    s = ctx.settings
    local_dir = s.local_base_dir()

    if ctx.dry_run:
        cached = local_dir.is_dir() and any(local_dir.iterdir())
        metrics = {
            "model": s.base_model_hf,
            "already_cached": cached,
            "target_dir": str(local_dir),
            "hf_token": "set" if s.resolved_hf_token() else "not set (fine for public models)",
        }
        note = "already cached locally" if cached else "would download from Hugging Face"
        return f"[dry] Base model {s.base_model_hf}, {note}.", metrics

    try:
        from huggingface_hub import snapshot_download  # lazy: training extra
    except ImportError as e:
        raise StageError("huggingface_hub is not installed. Run: pip install -e '.[train]'") from e

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(
            repo_id=s.base_model_hf, local_dir=local_dir, token=s.resolved_hf_token()
        )
    except Exception as e:  # hub errors are a zoo, the advice is not
        text = f"{type(e).__name__}: {e}"
        hint = _download_hint(text, s.base_model_hf, bool(s.resolved_hf_token()))
        raise StageError(f"Could not download {s.base_model_hf}. {text}.{hint}") from e

    return f"Pulled {s.base_model_hf}.", {"model": s.base_model_hf, "path": str(path)}
