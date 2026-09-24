"""Command line entry point.

Exists because the console is not always available. On a rented GPU box or in
a Colab runtime there is no browser pointed at the dev server, and you still
want to kick off a run and watch it in the terminal.

    finetune-lab run --real
    finetune-lab run --stages profile finetune
    finetune-lab plan
    finetune-lab serve
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .core.config import get_settings
from .core.logging import setup_logging
from .pipeline.runner import get_store, is_running, start_run
from .pipeline.status import STAGE_ORDER

POLL_S = 1.0


def _cmd_run(args: argparse.Namespace) -> int:
    dry = not args.real
    try:
        info = start_run(dry_run=dry, stages=args.stages or None)
    except Exception as e:  # a bad invocation deserves one line, not a traceback
        print(f"Could not start: {e}", file=sys.stderr)
        return 2

    mode = "dry" if dry else "real"
    print(f"Run {info['run_id']} ({mode}) in {info['run_dir']}")
    seen: set[str] = set()
    store = get_store()
    while True:
        status = store.read()
        # The worker thread writes the new run's first status a moment after
        # start_run returns. Until then the file still holds the previous run,
        # and printing that would report old results as this run's.
        if status.get("run_id") != info["run_id"]:
            if not is_running():
                print(
                    "\nRun ended before it wrote any status. Check the log above.", file=sys.stderr
                )
                return 1
            time.sleep(POLL_S / 10)
            continue
        for name in STAGE_ORDER:
            stage = status["stages"][name]
            if stage["state"] in ("done", "failed") and name not in seen:
                seen.add(name)
                mark = "ok  " if stage["state"] == "done" else "FAIL"
                print(f"  [{mark}] {stage['label']}: {stage['message']}")
        if not is_running() and status["state"] in ("done", "failed"):
            break
        time.sleep(POLL_S)

    if status["state"] == "failed":
        print(f"\nRun failed: {status['error']}", file=sys.stderr)
        return 1
    print("\nRun finished.")
    return 0


def _cmd_plan(_: argparse.Namespace) -> int:
    """Print what a real run would do on this machine, without running it."""
    from .pipeline.quantization import estimate_vram_gb, probe_backends, resolve_plan

    s = get_settings()
    backends = probe_backends()
    plan = resolve_plan(s, backends)
    print(
        json.dumps(
            {
                "base_model": s.base_model_hf,
                "hardware": backends.__dict__,
                "plan": plan.to_dict(),
                "vram_estimate": estimate_vram_gb(s.base_model_hf, plan, s.lora_r),
            },
            indent=2,
        )
    )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("finetune_lab.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="finetune-lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the pipeline and stream progress")
    p_run.add_argument("--real", action="store_true", help="real training instead of a dry run")
    p_run.add_argument("--stages", nargs="+", choices=STAGE_ORDER, help="subset of stages")
    p_run.set_defaults(func=_cmd_run)

    p_plan = sub.add_parser("plan", help="show the QLoRA plan for this machine")
    p_plan.set_defaults(func=_cmd_plan)

    p_serve = sub.add_parser("serve", help="start the API server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
