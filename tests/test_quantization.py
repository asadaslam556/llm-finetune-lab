"""The QLoRA strategy decision.

This is the module most likely to silently do the wrong thing, because the
wrong thing (falling back to LoRA) still trains a working adapter. So every
path here asserts on the reason string as well as the strategy: a downgrade
that nobody can see is the actual bug.
"""

import pytest

from finetune_lab.pipeline.quantization import (
    Backends,
    estimate_vram_gb,
    param_count_b,
    resolve_compute_dtype,
    resolve_plan,
)

# A machine that can genuinely do QLoRA.
GOOD_GPU = Backends(
    torch="2.5.1", cuda=True, gpu="NVIDIA RTX 4090", vram_gb=24.0, bf16=True, bitsandbytes="0.44.1"
)
# A pre-Ampere card: CUDA and bitsandbytes, but no bf16.
OLD_GPU = Backends(
    torch="2.5.1", cuda=True, gpu="NVIDIA Tesla T4", vram_gb=16.0, bf16=False, bitsandbytes="0.44.1"
)
NO_TORCH = Backends()
CPU_ONLY = Backends(torch="2.5.1", cuda=False)
GPU_NO_BNB = Backends(torch="2.5.1", cuda=True, gpu="RTX 3060", vram_gb=12.0, bf16=True)


class TestCanQuantize:
    def test_needs_both_cuda_and_bitsandbytes(self):
        assert GOOD_GPU.can_quantize is True
        assert GPU_NO_BNB.can_quantize is False
        # bitsandbytes imports fine on a CPU box and is then useless, so the
        # CUDA half of this matters as much as the package half.
        assert Backends(torch="2.5.1", cuda=False, bitsandbytes="0.44.1").can_quantize is False


class TestComputeDtype:
    def test_auto_prefers_bf16_where_available(self):
        assert resolve_compute_dtype("auto", GOOD_GPU) == "bfloat16"

    def test_auto_falls_back_to_fp16_on_older_cards(self):
        """Pre-Ampere has no bf16. Getting this wrong breaks training on
        exactly the GPUs most people have."""
        assert resolve_compute_dtype("auto", OLD_GPU) == "float16"

    def test_auto_is_fp32_without_cuda(self):
        assert resolve_compute_dtype("auto", CPU_ONLY) == "float32"

    def test_explicit_choice_is_respected(self):
        assert resolve_compute_dtype("float16", GOOD_GPU) == "float16"


class TestResolvePlan:
    def test_qlora_on_a_capable_machine(self, settings):
        plan = resolve_plan(settings, GOOD_GPU)
        assert plan.strategy == "qlora"
        assert plan.downgraded is False
        assert plan.bits == 4
        assert plan.quant_type == "nf4"
        assert plan.double_quant is True
        assert plan.compute_dtype == "bfloat16"
        assert plan.optimizer == "paged_adamw_8bit"

    @pytest.mark.parametrize(
        ("backends", "fragment"),
        [
            (NO_TORCH, "torch is not installed"),
            (CPU_ONLY, "No CUDA device"),
            (GPU_NO_BNB, "bitsandbytes is not installed"),
        ],
    )
    def test_downgrades_are_never_silent(self, settings, backends, fragment):
        plan = resolve_plan(settings, backends)
        assert plan.strategy == "lora"
        assert plan.downgraded is True
        assert fragment in plan.reason
        # The request is preserved so the UI can say "you asked for QLoRA".
        assert plan.requested == "qlora"

    def test_explicit_lora_is_not_a_downgrade(self, settings):
        settings.finetune_strategy = "lora"
        plan = resolve_plan(settings, GOOD_GPU)
        assert plan.strategy == "lora"
        assert plan.downgraded is False
        assert "explicitly" in plan.reason

    def test_paged_optimizer_dropped_without_bitsandbytes(self, settings):
        """paged_adamw_8bit is a bitsandbytes feature. Passing it through to
        the Trainer without bnb installed is an immediate crash."""
        plan = resolve_plan(settings, CPU_ONLY)
        assert plan.optimizer == "adamw_torch"

    def test_no_gradient_checkpointing_on_cpu(self, settings):
        """It trades compute for memory. On CPU there is no memory pressure
        to relieve and the compute is already the bottleneck."""
        assert resolve_plan(settings, CPU_ONLY).gradient_checkpointing is False

    def test_8bit_is_honoured(self, settings):
        settings.quant_bits = 8
        assert resolve_plan(settings, GOOD_GPU).bits == 8


class TestSizeHeuristics:
    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("Qwen/Qwen2.5-0.5B-Instruct", 0.5),
            ("Qwen/Qwen2.5-7B-Instruct", 7.0),
            ("meta-llama/Llama-3.2-1B-Instruct", 1.0),
            ("some/model-350M", 0.35),
        ],
    )
    def test_reads_param_count_off_the_repo_id(self, model_id, expected):
        assert param_count_b(model_id) == pytest.approx(expected)

    def test_unknown_shape_returns_none_rather_than_guessing(self):
        assert param_count_b("someone/a-model-with-no-size") is None

    def test_estimate_shows_qlora_saving_on_a_big_model(self, settings):
        plan = resolve_plan(settings, GOOD_GPU)
        est = estimate_vram_gb("Qwen/Qwen2.5-7B-Instruct", plan, settings.lora_r)
        assert est["qlora_gb"] < est["lora_bf16_gb"]
        assert est["saving_gb"] > 5  # 7B in 4-bit vs bf16 is roughly 10 GB back

    def test_saving_is_negligible_on_a_tiny_model(self, settings):
        """The honest result, and the reason the docs push 7B for QLoRA:
        at 0.5B the base weights were never what filled the card."""
        plan = resolve_plan(settings, GOOD_GPU)
        est = estimate_vram_gb("Qwen/Qwen2.5-0.5B-Instruct", plan, settings.lora_r)
        assert est["saving_gb"] < 1

    def test_no_estimate_when_size_is_unreadable(self, settings):
        plan = resolve_plan(settings, GOOD_GPU)
        assert estimate_vram_gb("someone/mystery-model", plan) is None
