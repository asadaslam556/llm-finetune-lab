"""Pipeline runner.

One background thread walks the requested stages in canonical order, timing
each one and pushing state into the atomic status file after every
transition. A failed stage stops the run, because later stages depend on
earlier artifacts and carrying on would just pile confusing errors on top of
the real one.

Single-flight by design: one training run at a time on one machine. A second
start attempt raises PipelineBusy, which the API turns into a 409.

Two things worth knowing about how runs use directories:

  * A run that includes `ingest` starts fresh in a new artifacts/<run_id>/.
  * A run that does not (say you only want to re-run `evaluate`) continues in
    the previous run's directory, because that is where its inputs live. If
    there is no previous directory, or the files it needs are not in it, the
    start is rejected up front with a message naming what is missing. Better
    than launching a run that is guaranteed to die three stages later.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

from ..core.config import get_settings
from ..core.errors import PipelineBusy
from ..core.logging import log
from .context import StageContext
from .stages import (
    s1_ingest,
    s2_prepare,
    s3_pull_base,
    s4_profile,
    s5_finetune,
    s6_evaluate,
    s7_export_deploy,
)
from .status import STAGE_INPUTS, STAGE_ORDER, STAGE_OUTPUTS, StatusStore

STAGE_FUNCS = {
    "ingest": s1_ingest.run,
    "prepare": s2_prepare.run,
    "pull_base": s3_pull_base.run,
    "profile": s4_profile.run,
    "finetune": s5_finetune.run,
    "evaluate": s6_evaluate.run,
    "export_deploy": s7_export_deploy.run,
}

# Inputs only a real run needs. Dry evaluate uses mock predictions and dry
# export just writes the Modelfile, so neither needs a trained adapter.
STAGE_INPUTS_REAL = {
    "evaluate": ["adapter"],
    "export_deploy": ["adapter"],
}

# A flag guarded by a lock, rather than a lock held for the whole run. Holding
# a lock across a thread boundary means releasing it from a thread that never
# acquired it, and one missed release wedges the app into permanent 409s until
# someone restarts it. A flag cleared in a finally block cannot get stuck.
_state_lock = threading.Lock()
_running = False


def get_store() -> StatusStore:
    return StatusStore(get_settings().artifacts_dir / "status.json")


def is_running() -> bool:
    with _state_lock:
        return _running


def _set_running(value: bool) -> None:
    global _running
    with _state_lock:
        _running = value


def _new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _latest_run_dir() -> Path | None:
    """Where the previous run put its files.

    Trusts the status file first, then falls back to the newest run-shaped
    directory on disk in case the status file got cleared.
    """
    settings = get_settings()
    recorded = get_store().read().get("run_dir")
    if recorded and Path(recorded).is_dir():
        return Path(recorded)
    candidates = [
        p for p in settings.artifacts_dir.glob("*-*") if p.is_dir() and p.name != "models"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _missing_inputs(stages: list[str], run_dir: Path, dry_run: bool) -> list[str]:
    """Which required files will not exist when each stage gets its turn."""
    produced: set[str] = set()
    missing: list[str] = []
    for name in stages:
        needed = list(STAGE_INPUTS[name])
        if not dry_run:
            needed += STAGE_INPUTS_REAL.get(name, [])
        for f in needed:
            if f not in produced and not (run_dir / f).exists():
                missing.append(f"{name} needs {f}")
        produced.update(STAGE_OUTPUTS[name])
    return missing


def _execute(run_id: str, dry_run: bool, stages: list[str], run_dir: Path) -> None:
    # Everything, including building the store, goes inside the try. Anything
    # left above it is code whose failure would skip the finally and leave the
    # busy flag stuck on.
    store = None
    status = None
    try:
        store = get_store()
        ctx = StageContext(settings=get_settings(), run_dir=run_dir, dry_run=dry_run)
        status = store.start_run(run_id, dry_run, stages, run_dir)
        for name in stages:
            store.stage_started(status, name)
            t0 = time.perf_counter()
            try:
                message, metrics = STAGE_FUNCS[name](ctx)
            except Exception as e:  # anything a stage throws ends the run
                log.exception("stage %s failed run_id=%s", name, run_id)
                store.stage_failed(status, name, time.perf_counter() - t0, str(e) or repr(e))
                return
            store.stage_finished(status, name, time.perf_counter() - t0, message, metrics)
        store.run_finished(status)
    except Exception:  # the worker itself broke, not a stage
        log.exception("pipeline runner crashed run_id=%s", run_id)
        # Best effort: leave something readable behind rather than a status
        # stuck on "running" forever.
        try:
            if store is None:
                return
            blob = status if status is not None else store.read()
            blob["state"] = "failed"
            blob["current_stage"] = None
            blob["finished_at"] = time.time()
            blob["error"] = "The pipeline runner crashed. Check the server log for the traceback."
            store.write(blob)
        except Exception:  # if even this fails, nothing left to try
            log.exception("could not record the runner crash")
    finally:
        _set_running(False)


def start_run(dry_run: bool = True, stages: list[str] | None = None) -> dict:
    """Validate, pick a directory, and hand off to a worker thread.

    Anything cheap to check gets checked here, in the caller's thread, so
    mistakes come back as a 4xx instead of a run that starts and then dies on
    its own a second later.
    """
    requested = stages or STAGE_ORDER
    unknown = [s for s in requested if s not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"Unknown stage(s): {', '.join(unknown)}. Known: {', '.join(STAGE_ORDER)}")
    chosen = [s for s in STAGE_ORDER if s in requested]
    if not chosen:
        raise ValueError(f"No valid stages in {stages}. Known: {', '.join(STAGE_ORDER)}")

    settings = get_settings()
    run_id = _new_run_id()
    if "ingest" in chosen:
        run_dir = settings.artifacts_dir / run_id
    else:
        previous = _latest_run_dir()
        if previous is None:
            raise ValueError(
                f"Stage(s) {', '.join(chosen)} continue an earlier run, but there is not one "
                "yet. Start a full run first, or include 'ingest'."
            )
        run_dir = previous

    missing = _missing_inputs(chosen, run_dir, dry_run)
    if missing:
        raise ValueError(
            f"Missing inputs in {run_dir}: {'; '.join(missing)}. "
            "Run the earlier stages first, or start a full run."
        )

    with _state_lock:
        global _running
        if _running:
            raise PipelineBusy("A run is already in progress. One at a time, it is one machine.")
        _running = True

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        thread = threading.Thread(
            target=_execute,
            args=(run_id, dry_run, chosen, run_dir),
            daemon=True,
            name=f"pipeline-{run_id}",
        )
        thread.start()
    except BaseException:
        # Do not leave the flag set just because we never got off the ground.
        _set_running(False)
        raise

    return {"run_id": run_id, "dry_run": dry_run, "stages": chosen, "run_dir": str(run_dir)}
