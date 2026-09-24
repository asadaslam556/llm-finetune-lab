"""Stage 6, evaluate.

The metrics are deliberately simple and explainable:

  * token-overlap F1 between the prediction and the reference answer
  * keyword hit rate, meaning does the prediction contain the reference's main
    content words (the "pin", "trash", "30 days" bits that make an answer
    correct rather than merely fluent)
  * average reply length, as a drift check

Dry mode gets predictions from the deterministic mock provider, so the report
is stable and the stage is unit-testable. Real mode loads the base plus your
adapter and generates against a validation sample.

For a QLoRA adapter the base is loaded in 4-bit here too. Inference under the
same quantization you trained under is both faster and more honest: it is the
configuration you are about to ship.
"""

from __future__ import annotations

import re
from collections import Counter

from ...core.errors import StageError
from ...providers.mock import MockProvider
from ..context import StageContext
from ..modeling import base_source, dtype_kwarg, load_tokenizer
from ..quantization import QuantPlan, bnb_config

_STOP = set(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "you",
        "your",
        "it",
        "this",
        "that",
        "with",
        "how",
        "do",
        "i",
        "can",
    ]
)

MAX_NEW_TOKENS = 128
EVAL_SAMPLE_CAP = 20  # keeps a real-mode eval in minutes, not hours


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def overlap_f1(pred: str, ref: str) -> float:
    p, r = Counter(_tokens(pred)), Counter(_tokens(ref))
    common = sum((p & r).values())
    if common == 0:
        return 0.0
    precision = common / max(1, sum(p.values()))
    recall = common / max(1, sum(r.values()))
    return round(2 * precision * recall / (precision + recall), 4)


def keyword_hit(pred: str, ref: str, k: int = 3) -> float:
    """Fraction of the reference's top content words that show up in the
    prediction.

    Counter.most_common keeps insertion order for ties, so this is stable
    across runs. That matters: a metric that wobbles between identical runs is
    worse than no metric.
    """
    content = [t for t in _tokens(ref) if t not in _STOP and len(t) > 2]
    if not content:
        return 1.0
    top = [t for t, _ in Counter(content).most_common()][:k]
    hits = sum(1 for t in top if t in _tokens(pred))
    return round(hits / len(top), 4)


def _predict_dry(rows: list[dict]) -> list[str]:
    mock = MockProvider()
    return [mock.chat(r["messages"][:-1]).reply for r in rows]


def _load_for_eval(ctx: StageContext, plan: QuantPlan):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    s = ctx.settings
    src = base_source(s)
    quant_config = bnb_config(plan)
    if quant_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            src, quantization_config=quant_config, device_map={"": 0}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(src, **dtype_kwarg("auto"))
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    adapter_dir = ctx.path("adapter")
    if not adapter_dir.is_dir():
        raise StageError(f"No adapter at {adapter_dir}. Run the fine-tune stage first.")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    # A quantized model is already placed by device_map, so read the device off
    # the model rather than guessing. Moving it again would undo the placement.
    return model, next(model.parameters()).device


def _predict_real(ctx: StageContext, rows: list[dict]) -> list[str]:
    try:
        import torch
    except ImportError as e:
        raise StageError("Training extras missing. Run: pip install -e '.[train]'") from e

    profile = ctx.read_json("profile.json", produced_by="profile")
    plan = QuantPlan(**profile["plan"])
    tok = load_tokenizer(ctx.settings)
    model, device = _load_for_eval(ctx, plan)

    preds = []
    try:
        for row in rows:
            prompt = tok.apply_chat_template(
                row["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
            ids = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            preds.append(tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise StageError(
                "Ran out of GPU memory generating eval samples. Free some VRAM, or evaluate on "
                "CPU with CUDA_VISIBLE_DEVICES=''."
            ) from e
        raise
    return preds


def run(ctx: StageContext) -> tuple[str, dict]:
    rows = ctx.read_jsonl("val.jsonl")
    sample = rows[:EVAL_SAMPLE_CAP]
    if not sample:
        raise StageError(
            "val.jsonl has no rows to evaluate. Re-run the prepare stage. With this little data "
            "the split may have produced an empty validation set."
        )

    preds = _predict_dry(sample) if ctx.dry_run else _predict_real(ctx, sample)
    refs = [r["messages"][-1]["content"] for r in sample]

    f1s = [overlap_f1(p, r) for p, r in zip(preds, refs, strict=True)]
    hits = [keyword_hit(p, r) for p, r in zip(preds, refs, strict=True)]
    report = {
        "mode": "simulated (mock predictions)" if ctx.dry_run else "real (adapter generations)",
        "n_eval": len(sample),
        "overlap_f1": round(sum(f1s) / len(f1s), 4),
        "keyword_hit_rate": round(sum(hits) / len(hits), 4),
        "avg_pred_chars": round(sum(len(p) for p in preds) / len(preds), 1),
        "examples": [
            {"question": row["messages"][1]["content"], "reference": ref, "prediction": pred}
            for row, ref, pred in list(zip(sample, refs, preds, strict=True))[:5]
        ],
    }
    ctx.write_json("eval_report.json", report)

    tag = "[dry] " if ctx.dry_run else ""
    return (
        f"{tag}Evaluated {len(sample)} samples: overlap-F1 {report['overlap_f1']}, "
        f"keyword hit rate {report['keyword_hit_rate']}.",
        {k: report[k] for k in ("mode", "n_eval", "overlap_f1", "keyword_hit_rate")},
    )
