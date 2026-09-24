"""Stage 1, ingest data.

Reads every .jsonl file in data/raw. Each row should be a support ticket:
{"id": ..., "question": ..., "answer": ..., "tag": ...}. Rows missing a
question or an answer get dropped and counted rather than crashed on. Real
support exports are always a little dirty, and one bad row should not kill a
run.

Same behaviour in dry and real mode, because reading local files is free.
"""

from __future__ import annotations

import json

from ...core.errors import StageError
from ..context import StageContext

REQUIRED = ("question", "answer")


def run(ctx: StageContext) -> tuple[str, dict]:
    raw_dir = ctx.settings.data_dir / "raw"
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise StageError(
            f"No .jsonl files in {raw_dir}. Nothing to train on. "
            "Generate the sample set with: python scripts/make_seed_data.py"
        )

    kept: list[dict] = []
    dropped = 0
    total = 0
    for fp in files:
        # utf-8-sig, not utf-8: Notepad and PowerShell's Out-File both write a
        # BOM by default, and a stray ﻿ makes json.loads reject the very
        # first ticket in the file. Took a confusing "why is one row always
        # missing" to find that one.
        try:
            with open(fp, encoding="utf-8-sig") as f:
                lines = f.readlines()
        except OSError as e:
            raise StageError(f"Could not read {fp}: {e}") from e

        for line in lines:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if not isinstance(row, dict):
                dropped += 1
                continue
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in REQUIRED):
                dropped += 1
                continue
            kept.append(row)

    if not kept:
        raise StageError(
            f"Every one of the {total} rows in {raw_dir} was unusable. Each line needs a "
            '"question" and an "answer" string.'
        )

    ctx.write_jsonl("ingested.jsonl", kept)
    metrics = {
        "files": len(files),
        "rows_seen": total,
        "rows_kept": len(kept),
        "rows_dropped": dropped,
    }
    return f"Ingested {len(kept)} tickets from {len(files)} file(s), dropped {dropped}.", metrics
