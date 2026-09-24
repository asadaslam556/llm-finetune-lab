"""One logger setup for the whole app.

Nothing clever. A single stdout handler with a readable format, guarded so
that re-importing the app in tests does not stack handlers and print every
line five times.
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"

log = logging.getLogger("finetune_lab")


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)


def mask(secret: str | None, keep_head: int = 6, keep_tail: int = 4) -> str:
    """Render a credential safely for logs and health output.

    Health responses get pasted into bug reports and screenshots, so nothing
    in this project ever prints a whole key.
    """
    if not secret:
        return "not set"
    if len(secret) <= keep_head + keep_tail:
        return "set (too short to mask, check it)"
    return f"{secret[:keep_head]}...{secret[-keep_tail:]}"
