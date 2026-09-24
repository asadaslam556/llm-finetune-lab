"""Mock provider.

Deterministic, instant, zero dependencies. The test suite uses it, and it is
handy in the UI when you want to click around the console with nothing
installed and nothing running.

Replies are keyed off a hash of the last user message, so the same question
always gets the same answer. The dry-run evaluation stage depends on that.
"""

from __future__ import annotations

import hashlib
import time

from .base import ChatResult, Provider

_CANNED = [
    "You can pin a note in Nimbus by long-pressing it and choosing Pin to top.",
    "Nimbus syncs every 30 seconds while you are online; offline edits merge on reconnect.",
    "To share a notebook, open it, tap Share, and pick view-only or can-edit.",
    "Deleted notes sit in Trash for 30 days before Nimbus removes them for good.",
    "You can export any note as Markdown or PDF from the note's ... menu.",
]


class MockProvider(Provider):
    name = "mock"
    note = "Canned deterministic replies. For tests and clicking around with nothing installed."

    def chat(self, messages: list[dict], model: str | None = None) -> ChatResult:
        start = time.perf_counter()
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        idx = int(hashlib.sha256(last_user.encode()).hexdigest(), 16) % len(_CANNED)
        return ChatResult(
            reply=_CANNED[idx],
            model=model or self.default_model(),
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def default_model(self) -> str:
        return "mock-nimbus-v0"
