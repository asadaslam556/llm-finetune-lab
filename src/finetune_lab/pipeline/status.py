"""Pipeline status, shared as a single JSON file.

Why a file instead of in-memory state: the runner works in a background
thread, the API reads from request handlers, and eventually you might want a
separate training process entirely. A JSON file everyone can read is the
dumbest thing that works. The write is atomic (temp file plus os.replace), so
a reader can never catch it half written.

The Windows wrinkle: os.replace is atomic there too, but it throws a sharing
violation if another handle has the destination open at that instant. Our own
status poller has it open every 1.5 seconds, so this is not hypothetical. The
write retries a few times, and the read treats any OSError as "nothing to
report yet" rather than letting it surface as a 500.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

STAGE_ORDER = [
    "ingest",
    "prepare",
    "pull_base",
    "profile",
    "finetune",
    "evaluate",
    "export_deploy",
]

# Short labels. They live in a narrow left column.
STAGE_LABELS = {
    "ingest": "Ingest data",
    "prepare": "Prepare & format",
    "pull_base": "Pull base model",
    "profile": "Profile & plan",
    "finetune": "Fine-tune (QLoRA)",
    "evaluate": "Evaluate",
    "export_deploy": "Export & deploy",
}

# Files each stage needs before it can do anything. Used to reject a subset
# run up front with a useful message instead of failing three stages in with
# a bare "No such file or directory".
STAGE_INPUTS = {
    "ingest": [],
    "prepare": ["ingested.jsonl"],
    "pull_base": [],
    "profile": ["train.jsonl"],
    "finetune": ["train.jsonl", "profile.json"],
    "evaluate": ["val.jsonl"],
    "export_deploy": [],
}

# ...and what each one leaves behind, so we can tell whether an earlier stage
# in the same run will satisfy a later stage's needs.
STAGE_OUTPUTS = {
    "ingest": ["ingested.jsonl"],
    "prepare": ["train.jsonl", "val.jsonl"],
    "pull_base": [],
    "profile": ["profile.json"],
    "finetune": ["adapter"],
    "evaluate": ["eval_report.json"],
    "export_deploy": ["Modelfile", "DEPLOY_PLAN.md"],
}

_write_lock = threading.Lock()

_REPLACE_ATTEMPTS = 6
_REPLACE_BACKOFF_S = 0.02


def _blank(
    run_id: str | None = None,
    dry_run: bool = False,
    stages: list[str] | None = None,
) -> dict[str, Any]:
    chosen = stages or STAGE_ORDER
    return {
        "run_id": run_id,
        "state": "idle",  # idle | running | done | failed
        "dry_run": dry_run,
        "current_stage": None,
        "run_dir": None,
        "started_at": None,
        "finished_at": None,
        "error": None,
        "stages": {
            name: {
                "label": STAGE_LABELS[name],
                "state": "pending" if name in chosen else "skipped",
                "started_at": None,
                "seconds": None,
                "message": "",
                "metrics": {},
            }
            for name in STAGE_ORDER
        },
    }


class StatusStore:
    def __init__(self, path: Path):
        self.path = path

    def read(self) -> dict[str, Any]:
        """Never raises. Missing, half written, hand-mangled or briefly locked
        all mean the same thing to a caller: nothing useful to report yet."""
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return _blank()
        if not isinstance(data, dict) or "stages" not in data:
            return _blank()
        return data

    def write(self, status: dict[str, Any]) -> None:
        with _write_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                self._replace_with_retry(tmp)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)

    def _replace_with_retry(self, tmp: str) -> None:
        """os.replace, but patient about Windows sharing violations."""
        last = None
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError as e:  # a reader has it open this instant
                last = e
                time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))
        raise OSError(
            f"Could not update {self.path} after {_REPLACE_ATTEMPTS} tries. "
            "Something is holding it open."
        ) from last

    # -- helpers the runner uses --

    def start_run(
        self, run_id: str, dry_run: bool, stages: list[str], run_dir: Path
    ) -> dict[str, Any]:
        status = _blank(run_id, dry_run, stages)
        status["state"] = "running"
        status["started_at"] = time.time()
        status["run_dir"] = str(run_dir)
        self.write(status)
        return status

    def stage_started(self, status: dict[str, Any], name: str) -> None:
        status["current_stage"] = name
        status["stages"][name]["state"] = "running"
        status["stages"][name]["started_at"] = time.time()
        self.write(status)

    def stage_finished(
        self,
        status: dict[str, Any],
        name: str,
        seconds: float,
        message: str,
        metrics: dict[str, Any],
    ) -> None:
        s = status["stages"][name]
        s.update(state="done", seconds=round(seconds, 2), message=message, metrics=metrics)
        self.write(status)

    def stage_failed(self, status: dict[str, Any], name: str, seconds: float, error: str) -> None:
        s = status["stages"][name]
        s.update(state="failed", seconds=round(seconds, 2), message=error)
        status["state"] = "failed"
        status["current_stage"] = None
        status["error"] = f"{STAGE_LABELS[name]}: {error}"
        status["finished_at"] = time.time()
        self.write(status)

    def run_finished(self, status: dict[str, Any]) -> None:
        status["state"] = "done"
        status["current_stage"] = None
        status["finished_at"] = time.time()
        self.write(status)
