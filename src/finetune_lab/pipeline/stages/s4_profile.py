"""Stage 4, profile the machine and write a training plan.

Looks at the hardware, decides QLoRA or LoRA, and writes a concrete plan:
device, quantization, compute dtype, per-device batch size, gradient
accumulation and estimated optimizer steps. The training stage reads this
plan instead of re-deciding anything, so what you see on the rail is exactly
what training will do.

Runs the same in dry mode, because poking at the hardware is free. That is
deliberate: a dry run on your laptop tells you honestly whether a real QLoRA
run would work there, before you spend twenty minutes on a download.
"""

from __future__ import annotations

import math
import os
import platform

from ...core.errors import StageError
from ..context import StageContext
from ..quantization import estimate_vram_gb, probe_backends, resolve_plan


def _batch_plan(settings, backends, plan) -> tuple[int, int]:
    """Per-device batch size and gradient accumulation.

    The effective batch stays roughly what the user asked for; only the split
    between the two changes. 4-bit weights leave more room for activations, so
    QLoRA can afford a bigger per-device batch on the same card than LoRA can.
    """
    target = max(1, settings.batch_size)
    if not backends.cuda:
        # CPU training is for smoke tests only. Keep it honest and small.
        return 1, max(1, target)

    vram = backends.vram_gb or 0
    headroom = vram * (2.0 if plan.strategy == "qlora" else 1.0)
    if headroom >= 16:
        batch = target
    elif headroom >= 8:
        batch = max(1, target // 2)
    else:
        batch = 1
    grad_accum = max(1, round(target / batch))
    return batch, grad_accum


def run(ctx: StageContext) -> tuple[str, dict]:
    s = ctx.settings
    backends = probe_backends()
    plan = resolve_plan(s, backends)
    batch, grad_accum = _batch_plan(s, backends, plan)

    rows = ctx.read_jsonl("train.jsonl") if ctx.path("train.jsonl").exists() else []
    if not rows:
        raise StageError("train.jsonl is empty or missing. Run the prepare stage first.")
    steps = math.ceil(len(rows) * s.num_epochs / max(1, batch * grad_accum))

    report = {
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "device": "cuda" if backends.cuda else "cpu",
        "hardware": {
            "torch": backends.torch,
            "cuda": backends.cuda,
            "gpu": backends.gpu,
            "vram_gb": backends.vram_gb,
            "bf16": backends.bf16,
            "bitsandbytes": backends.bitsandbytes,
        },
        "plan": plan.to_dict(),
        "lora": {
            "r": s.lora_r,
            "alpha": s.lora_alpha,
            "dropout": s.lora_dropout,
            "target_modules": s.lora_target_modules,
        },
        "vram_estimate": estimate_vram_gb(s.base_model_hf, plan, s.lora_r),
        "per_device_batch": batch,
        "grad_accum": grad_accum,
        "effective_batch": batch * grad_accum,
        "epochs": s.num_epochs,
        "train_rows": len(rows),
        "estimated_steps": steps,
    }
    ctx.write_json("profile.json", report)

    gpu_bit = f"{backends.gpu} ({backends.vram_gb} GB)" if backends.gpu else "no GPU detected"
    strategy_bit = plan.strategy.upper()
    if plan.downgraded:
        strategy_bit += " (downgraded from QLoRA)"
    message = (
        f"{strategy_bit} on {report['device']}/{plan.compute_dtype}, "
        f"batch {batch}x{grad_accum}, ~{steps} steps. {gpu_bit}. {plan.reason}"
    )
    metrics = {
        "strategy": plan.strategy,
        "downgraded": plan.downgraded,
        "reason": plan.reason,
        "device": report["device"],
        "compute_dtype": plan.compute_dtype,
        "effective_batch": batch * grad_accum,
        "estimated_steps": steps,
        "vram_estimate": report["vram_estimate"],
    }
    return message, metrics
