"""QLoRA strategy resolution.

This is the module the whole QLoRA migration hangs off. It answers one
question: given what the user asked for and what this machine can actually
do, do we load the base model in 4-bit or not, and why?

Keeping the decision here instead of inside the training stage has two
payoffs. The profile stage can report the plan before anything heavy runs,
and the health endpoint can tell you whether 4-bit would work on this box
without importing torch into the API process.

Worth stating plainly, because it is the thing people get wrong: QLoRA is
not "better LoRA". It is LoRA with the frozen base model quantized to 4-bit.
You buy a large VRAM saving and you pay for it in step time, because every
forward pass dequantizes weights on the fly. On a 7B model that trade is
excellent. On a 0.5B model the base weights were never the problem, so you
pay the speed cost for a saving you did not need. docs/qlora.md has the
numbers.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# Bytes per parameter for the frozen base weights, by how we load them.
_BYTES_PER_PARAM = {4: 0.5, 8: 1.0, 16: 2.0, 32: 4.0}

# Rough parameter counts we can read straight off a Hugging Face repo id.
# ponytail: string heuristic, good enough for a VRAM estimate in a report.
# If you need it exact, read config.json after the pull stage.
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([bm])\b", re.IGNORECASE)


@dataclass
class Backends:
    """What the training stack on this machine can actually do."""

    torch: str | None = None
    cuda: bool = False
    gpu: str | None = None
    vram_gb: float | None = None
    bf16: bool = False
    bitsandbytes: str | None = None

    @property
    def can_quantize(self) -> bool:
        # bitsandbytes needs a CUDA device for the fast 4-bit kernels. It will
        # import happily on a CPU-only box and then be useless, so both halves
        # of this matter.
        return bool(self.cuda and self.bitsandbytes)


@dataclass
class QuantPlan:
    """The decision, plus the reason, so the UI never has to guess."""

    strategy: str  # "qlora" | "lora"
    requested: str
    bits: int | None  # None when we are not quantizing
    quant_type: str | None
    double_quant: bool
    compute_dtype: str  # bfloat16 | float16 | float32
    gradient_checkpointing: bool
    optimizer: str
    reason: str
    downgraded: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def probe_backends() -> Backends:
    """Look at the machine. Never raises, because a failed probe should read
    as "no GPU found" rather than kill a run before it starts."""
    try:
        import torch
    except ImportError:
        return Backends()

    b = Backends(torch=torch.__version__)
    # A torch install with a mismatched driver can throw from any of these.
    try:
        b.cuda = torch.cuda.is_available()
        if b.cuda:
            b.gpu = torch.cuda.get_device_name(0)
            b.vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            b.bf16 = bool(torch.cuda.is_bf16_supported())
    except Exception:  # a driver mismatch can throw from any of these
        b.cuda, b.gpu, b.vram_gb, b.bf16 = False, None, None, False

    try:
        import bitsandbytes

        b.bitsandbytes = getattr(bitsandbytes, "__version__", "unknown")
    except Exception:  # bnb raises plenty that is not ImportError
        b.bitsandbytes = None
    return b


def resolve_compute_dtype(requested: str, backends: Backends) -> str:
    """bf16 where the hardware has it, fp16 on older CUDA cards, fp32 on CPU.

    bf16 is the one you want: same exponent range as fp32, so no loss scaling
    and no overflow drama. Ampere and newer have it. Anything older (GTX 16xx,
    RTX 20xx, T4, V100) gets fp16 instead, which is why the training stage
    still carries fp16 handling.
    """
    if requested != "auto":
        return requested
    if not backends.cuda:
        return "float32"
    return "bfloat16" if backends.bf16 else "float16"


def resolve_plan(settings, backends: Backends) -> QuantPlan:
    """Pick the strategy we will actually run, and say why.

    QLoRA is the default. It downgrades to plain LoRA rather than failing when
    the machine cannot do 4-bit, because a dry run on a laptop and a CI job on
    a CPU runner both have to keep working. Every downgrade carries a reason
    string that ends up on the pipeline rail, so it is never silent.
    """
    requested = settings.finetune_strategy
    dtype = resolve_compute_dtype(settings.compute_dtype, backends)

    def lora(reason: str, downgraded: bool) -> QuantPlan:
        return QuantPlan(
            strategy="lora",
            requested=requested,
            bits=None,
            quant_type=None,
            double_quant=False,
            compute_dtype=dtype,
            # Checkpointing costs ~30% speed. Only worth it on a real GPU.
            gradient_checkpointing=settings.gradient_checkpointing and backends.cuda,
            # Paged optimizers are a bitsandbytes feature. Without it, plain
            # adamw, otherwise the trainer dies on an unknown optim string.
            optimizer=settings.optimizer if backends.bitsandbytes else "adamw_torch",
            reason=reason,
            downgraded=downgraded,
        )

    if requested == "lora":
        return lora("LoRA requested explicitly (LFL_FINETUNE_STRATEGY=lora).", False)

    if not backends.torch:
        return lora(
            "torch is not installed, so nothing can be quantized. Install the training "
            "extras: pip install -e '.[train]'",
            True,
        )
    if not backends.cuda:
        return lora(
            "No CUDA device. 4-bit kernels need one, so this falls back to LoRA. "
            "Dry runs are unaffected; for a real run use a GPU box or the Colab notebook.",
            True,
        )
    if not backends.bitsandbytes:
        return lora(
            "bitsandbytes is not installed, which is what provides 4-bit. "
            "Install it: pip install bitsandbytes",
            True,
        )

    return QuantPlan(
        strategy="qlora",
        requested=requested,
        bits=settings.quant_bits,
        quant_type=settings.quant_type,
        double_quant=settings.double_quant,
        compute_dtype=dtype,
        gradient_checkpointing=settings.gradient_checkpointing,
        optimizer=settings.optimizer,
        reason=(
            f"{settings.quant_bits}-bit {settings.quant_type} on {backends.gpu} "
            f"via bitsandbytes {backends.bitsandbytes}, compute in {dtype}."
        ),
        downgraded=False,
    )


def bnb_config(plan: QuantPlan):
    """Build the transformers BitsAndBytesConfig for this plan.

    Lazy import so the API server never drags in torch. Returns None for the
    LoRA path, which is what from_pretrained wants when it should not quantize.
    """
    if plan.strategy != "qlora":
        return None

    import torch
    from transformers import BitsAndBytesConfig

    compute = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        plan.compute_dtype, torch.float32
    )
    if plan.bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=compute)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=plan.quant_type,
        bnb_4bit_use_double_quant=plan.double_quant,
        bnb_4bit_compute_dtype=compute,
    )


def param_count_b(model_id: str) -> float | None:
    """Guess parameter count in billions from a repo id like Qwen2.5-7B-Instruct.

    Used only for the VRAM estimate in the profile report, so a miss costs a
    slightly wrong number in a table and nothing else.
    """
    for value, unit in _SIZE_RE.findall(model_id):
        n = float(value)
        return n if unit.lower() == "b" else n / 1000
    return None


def estimate_vram_gb(model_id: str, plan: QuantPlan, lora_r: int = 16) -> dict | None:
    """Ballpark VRAM for this plan, and for the alternative, side by side.

    Deliberately rough. Base weights plus adapters plus optimizer state, with
    a flat allowance for activations and fragmentation. It exists to answer
    "will this fit and is quantizing worth it", not to be precise.
    """
    b = param_count_b(model_id)
    if b is None:
        return None
    params = b * 1e9

    def weights_gb(bits: int) -> float:
        return params * _BYTES_PER_PARAM[bits] / 1024**3

    # LoRA adds roughly 2 * r * hidden per targeted matrix. At the ranks used
    # here it lands well under 1% of the base, so a flat fraction is honest
    # enough for a ballpark and far less fragile than guessing layer shapes.
    adapter_gb = params * 0.004 * (lora_r / 16) * 2 / 1024**3
    optim_gb = adapter_gb * (2 if "8bit" in plan.optimizer else 8)
    overhead_gb = 1.5 if plan.gradient_checkpointing else 3.0

    quantized = weights_gb(plan.bits or 16) + adapter_gb + optim_gb + overhead_gb
    unquantized = weights_gb(16) + adapter_gb + optim_gb + overhead_gb
    return {
        "params_b": b,
        "qlora_gb": round(quantized, 1),
        "lora_bf16_gb": round(unquantized, 1),
        "saving_gb": round(unquantized - quantized, 1),
        "note": "Estimate only. Activations vary with batch size and sequence length.",
    }
