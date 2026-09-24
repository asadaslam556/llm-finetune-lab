"""What every stage gets handed.

Stages are plain functions: run(ctx) -> (message, metrics). No classes, no
framework. A stage that is just a function is trivially unit-testable, and
the runner stays a short loop instead of an orchestration engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..core.config import Settings
from ..core.errors import StageError


@dataclass
class StageContext:
    settings: Settings
    run_dir: Path  # artifacts/<run_id>/, where every file a stage writes goes
    dry_run: bool

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def read_json(self, name: str, produced_by: str) -> dict:
        """Load an artifact an earlier stage wrote, or explain what is missing.

        Every stage that depends on an upstream file went through the same
        three lines of try/except before this existed.
        """
        try:
            with open(self.path(name), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise StageError(
                f"{name} is missing or unreadable. Run the {produced_by} stage first."
            ) from e

    def read_jsonl(self, name: str) -> list[dict]:
        # utf-8-sig, not utf-8: Notepad and PowerShell's Out-File both write a
        # BOM, and a stray ﻿ makes json.loads reject the first row.
        with open(self.path(name), encoding="utf-8-sig") as f:
            return [json.loads(line) for line in f if line.strip()]

    def write_json(self, name: str, data: dict) -> None:
        with open(self.path(name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def write_jsonl(self, name: str, rows: list[dict]) -> None:
        with open(self.path(name), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
