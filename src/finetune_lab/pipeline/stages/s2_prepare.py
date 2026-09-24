"""Stage 2, prepare and format.

Turns raw tickets into chat-format training rows:

  {"messages": [{"role": "system", ...}, {"role": "user", ...},
                {"role": "assistant", ...}]}

That is the shape the tokenizer's chat template expects in the training
stage, and it is also what most SFT tooling reads, so the prepared files are
reusable outside this repo.

Cleaning rules, in order:
  1. dedupe on the normalised question (lowercased, whitespace squashed),
     because support exports love exact duplicates
  2. drop answers shorter than min_answer_chars, since "yes." teaches nothing
  3. deterministic shuffle on settings.seed, then train/val split

All pure local compute, so dry and real mode are identical.
"""

from __future__ import annotations

import random
import re

from ...core.errors import StageError
from ..context import StageContext

SYSTEM_PROMPT = (
    "You are the support assistant for Nimbus, a note-taking app. "
    "Answer clearly and concisely using only Nimbus features."
)


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def to_chat_row(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"].strip()},
            {"role": "assistant", "content": row["answer"].strip()},
        ]
    }


def run(ctx: StageContext) -> tuple[str, dict]:
    rows = ctx.read_jsonl("ingested.jsonl")

    seen: set[str] = set()
    deduped: list[dict] = []
    dupes = 0
    for row in rows:
        key = _norm(row["question"])
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        deduped.append(row)

    min_chars = ctx.settings.min_answer_chars
    filtered = [r for r in deduped if len(r["answer"].strip()) >= min_chars]
    too_short = len(deduped) - len(filtered)

    if len(filtered) < 10:
        raise StageError(
            f"Only {len(filtered)} usable rows after cleaning. That is not a dataset. "
            "Add more tickets to data/raw, or lower LFL_MIN_ANSWER_CHARS."
        )

    rng = random.Random(ctx.settings.seed)
    rng.shuffle(filtered)
    n_val = max(1, int(len(filtered) * ctx.settings.val_split))
    val, train = filtered[:n_val], filtered[n_val:]

    ctx.write_jsonl("train.jsonl", [to_chat_row(r) for r in train])
    ctx.write_jsonl("val.jsonl", [to_chat_row(r) for r in val])

    metrics = {
        "duplicates_dropped": dupes,
        "too_short_dropped": too_short,
        "train_rows": len(train),
        "val_rows": len(val),
    }
    return (
        f"Prepared {len(train)} train / {len(val)} val rows "
        f"(dropped {dupes} dupes, {too_short} too-short).",
        metrics,
    )
