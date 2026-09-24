"""Stage 5, fine-tune with QLoRA.

Dry mode simulates training. It writes the exact adapter and quantization
config a real run would use, plus a deterministic fake loss curve, so the UI
and the tests can exercise the whole flow with no GPU and no downloads. The
simulation is labelled as such everywhere it appears. Nothing here ever
pretends a fake loss is a real one.

Real mode is the QLoRA recipe:

  1. load the frozen base in 4-bit NF4 through bitsandbytes
  2. prepare_model_for_kbit_training, which upcasts the layer norms and the
     output head back to fp32 and switches on gradient checkpointing
  3. attach LoRA adapters to every linear layer, in bf16 or fp16
  4. train only the adapters with a paged optimizer

Steps 1 and 2 are the whole difference from plain LoRA. The base weights sit
in 4-bit and never get gradients; the small adapters carry all the learning
in higher precision. Step 2 is not optional: without the fp32 upcast the
loss goes to NaN within a few dozen steps on fp16 hardware.

The strategy actually used comes from profile.json, which means a machine
that cannot do 4-bit degrades to LoRA rather than failing. That decision, and
its reason, is made in pipeline/quantization.py.
"""

from __future__ import annotations

import math
import time

from ...core.errors import StageError
from ...core.logging import log
from ..context import StageContext
from ..modeling import base_source, dtype_kwarg, load_tokenizer
from ..quantization import QuantPlan, bnb_config


def _plan_from_profile(ctx: StageContext) -> QuantPlan:
    profile = ctx.read_json("profile.json", produced_by="profile")
    try:
        return QuantPlan(**profile["plan"])
    except (KeyError, TypeError) as e:
        raise StageError("profile.json has no usable plan in it. Re-run the profile stage.") from e


def _adapter_report(s, plan: QuantPlan) -> dict:
    """Everything needed to reproduce this run, written next to the adapter."""
    return {
        "base_model": s.base_model_hf,
        "strategy": plan.strategy,
        "quantization": (
            {
                "bits": plan.bits,
                "quant_type": plan.quant_type,
                "double_quant": plan.double_quant,
                "compute_dtype": plan.compute_dtype,
            }
            if plan.strategy == "qlora"
            else None
        ),
        "lora": {
            "r": s.lora_r,
            "alpha": s.lora_alpha,
            "dropout": s.lora_dropout,
            "target_modules": s.lora_target_modules,
        },
        "trainer": {
            "learning_rate": s.learning_rate,
            "epochs": s.num_epochs,
            "max_seq_len": s.max_seq_len,
            "optimizer": plan.optimizer,
            "lr_scheduler": s.lr_scheduler,
            "warmup_ratio": s.warmup_ratio,
            "gradient_checkpointing": plan.gradient_checkpointing,
            "seed": s.seed,
        },
    }


IGNORE_INDEX = -100  # what the loss function skips


def build_example(tokenizer, messages: list[dict], max_len: int) -> dict:
    """Token ids for one conversation, with labels that only score the reply.

    Everything before the assistant turn (system prompt, user question) gets
    IGNORE_INDEX, so the model is graded on answering, not on reciting the
    system prompt it sees in every single row. The rendered prompt is a text
    prefix of the rendered conversation, so its token count marks where the
    answer starts. add_special_tokens=False because the chat template already
    wrote them; a second BOS would shift everything by one.
    """
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_len]
    cut = min(prompt_len, len(ids))
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [IGNORE_INDEX] * cut + ids[cut:],
    }


def _simulated_losses(steps: int, seed: int) -> list[float]:
    # Smooth-ish exponential decay with a little seeded wobble. Looks like a
    # loss curve, is clearly not one. The report says "simulated" on it.
    import random

    rng = random.Random(seed)
    return [
        round(2.4 * math.exp(-3 * i / max(1, steps)) + rng.uniform(0.02, 0.09) + 0.35, 4)
        for i in range(steps)
    ]


def _run_dry(ctx: StageContext, plan: QuantPlan, steps: int) -> tuple[str, dict]:
    s = ctx.settings
    adapter_dir = ctx.path("adapter")
    losses = _simulated_losses(min(steps, 200), s.seed)
    report = {
        "mode": "simulated",
        "config": _adapter_report(s, plan),
        "steps": len(losses),
        "loss_curve": losses,
        "final_loss": losses[-1],
    }
    ctx.write_json("adapter/ADAPTER_INFO.json", report)
    time.sleep(0.4)  # so the rail visibly hits "running" in the UI demo
    label = plan.strategy.upper()
    return (
        f"[dry] Simulated {len(losses)} {label} steps, final loss {losses[-1]} (not a real loss).",
        {
            "mode": "simulated",
            "strategy": plan.strategy,
            "steps": len(losses),
            "final_loss": losses[-1],
            "adapter_dir": str(adapter_dir),
        },
    )


def _build_model(ctx: StageContext, plan: QuantPlan):
    """Load the base under this plan and attach LoRA adapters."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    s = ctx.settings
    src = base_source(s)
    quant_config = bnb_config(plan)

    if quant_config is not None:
        # device_map pins the whole model to GPU 0. bitsandbytes cannot train
        # layers that accelerate has offloaded to CPU or disk, and the failure
        # is an obscure dtype error twenty minutes in, so pin it up front.
        model = AutoModelForCausalLM.from_pretrained(
            src,
            quantization_config=quant_config,
            device_map={"": 0},
            **dtype_kwarg(torch.bfloat16 if plan.compute_dtype == "bfloat16" else torch.float16),
        )
        # This is the step that makes 4-bit training stable: layer norms and
        # the output head go back to fp32, inputs get requires_grad so
        # checkpointing has something to hook, and use_cache goes off.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=plan.gradient_checkpointing
        )
    else:
        # Plain LoRA, and the one trap in this whole file:
        #
        # bf16 -> load the weights in bf16 and train in bf16. No scaler needed.
        # fp16 -> load the weights in fp32 and let the Trainer's AMP do the
        #         casting. Loading in fp16 and also passing fp16=True gives you
        #         "Attempting to unscale FP16 gradients" the moment the grad
        #         scaler runs, because there are no fp32 master weights left to
        #         unscale into. Every pre-Ampere card takes this branch, so
        #         getting it wrong breaks training on exactly the GPUs most
        #         people have.
        load_dtype = torch.bfloat16 if plan.compute_dtype == "bfloat16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(src, **dtype_kwarg(load_dtype))

    if plan.gradient_checkpointing:
        # use_reentrant=False is required with PEFT. The reentrant version
        # cannot see that only the adapters need gradients and throws
        # "none of the inputs have requires_grad" on the first backward pass.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            r=s.lora_r,
            lora_alpha=s.lora_alpha,
            lora_dropout=s.lora_dropout,
            target_modules=s.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    return model


def _trainable_summary(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / max(1, total), 4),
    }


def _run_real(ctx: StageContext, plan: QuantPlan) -> tuple[str, dict]:
    try:
        import torch
        from datasets import load_dataset
        from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    except ImportError as e:
        raise StageError(
            "Training extras missing. Run: pip install -e '.[train]' "
            "(and pip install bitsandbytes for the QLoRA path)."
        ) from e

    s = ctx.settings
    profile = ctx.read_json("profile.json", produced_by="profile")
    adapter_dir = ctx.path("adapter")
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(s)
    model = _build_model(ctx, plan)
    sizes = _trainable_summary(model)
    log.info(
        "%s: training %s of %s params (%.4f%%)",
        plan.strategy,
        f"{sizes['trainable_params']:,}",
        f"{sizes['total_params']:,}",
        sizes["trainable_pct"],
    )

    ds = load_dataset("json", data_files=str(ctx.path("train.jsonl")), split="train")
    ds = ds.map(
        lambda ex: build_example(tokenizer, ex["messages"], s.max_seq_len),
        remove_columns=ds.column_names,
    )
    # A prompt longer than max_seq_len leaves no answer tokens to learn from,
    # and a batch of only those gives a NaN loss.
    ds = ds.filter(lambda ex: any(label != IGNORE_INDEX for label in ex["labels"]))
    if len(ds) == 0:
        raise StageError(
            f"No training rows fit in LFL_MAX_SEQ_LEN={s.max_seq_len} with room for the "
            "answer. Raise it."
        )

    args = TrainingArguments(
        output_dir=str(ctx.path("trainer_out")),
        per_device_train_batch_size=profile["per_device_batch"],
        gradient_accumulation_steps=profile["grad_accum"],
        num_train_epochs=s.num_epochs,
        learning_rate=s.learning_rate,
        lr_scheduler_type=s.lr_scheduler,
        warmup_ratio=s.warmup_ratio,
        optim=plan.optimizer,
        logging_steps=5,
        save_strategy="no",  # we save the adapter ourselves below
        seed=s.seed,
        report_to=[],
        bf16=plan.compute_dtype == "bfloat16",
        fp16=plan.compute_dtype == "float16",
        gradient_checkpointing=False,  # already enabled on the model itself
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        # Seq2Seq's collator pads labels with IGNORE_INDEX and keeps our
        # masking. The language-modeling one would overwrite labels with a
        # copy of input_ids and undo it.
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX),
    )

    try:
        result = trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        extra = (
            ""
            if plan.strategy == "qlora"
            else " You are on the LoRA path; QLoRA would cut base-weight memory by about 75%."
        )
        raise StageError(
            f"GPU ran out of memory at batch size {profile['per_device_batch']}. Lower it with "
            f"LFL_BATCH_SIZE=1, or shorten sequences with LFL_MAX_SEQ_LEN=256.{extra}"
        ) from e
    except Exception as e:  # surface the reason, not a traceback
        raise StageError(f"Training stopped: {type(e).__name__}: {e}") from e

    # Saves adapter weights only, a few megabytes rather than the whole model.
    # That is the other half of why QLoRA is cheap: you ship the delta.
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    final_loss = round(result.training_loss, 4)
    ctx.write_json(
        "adapter/ADAPTER_INFO.json",
        {
            "mode": "real",
            "config": _adapter_report(s, plan),
            "steps": int(result.global_step),
            "final_loss": final_loss,
            **sizes,
        },
    )
    return (
        f"Trained {result.global_step} {plan.strategy.upper()} steps, final loss {final_loss}. "
        f"Adapter saved ({sizes['trainable_pct']}% of params trainable).",
        {
            "mode": "real",
            "strategy": plan.strategy,
            "steps": int(result.global_step),
            "final_loss": final_loss,
            **sizes,
        },
    )


def run(ctx: StageContext) -> tuple[str, dict]:
    plan = _plan_from_profile(ctx)
    ctx.path("adapter").mkdir(parents=True, exist_ok=True)
    if ctx.dry_run:
        profile = ctx.read_json("profile.json", produced_by="profile")
        return _run_dry(ctx, plan, max(1, profile["estimated_steps"]))
    return _run_real(ctx, plan)
