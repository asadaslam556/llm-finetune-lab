"""Error types shared across layers.

Kept deliberately thin. The API layer maps these to status codes, and
everything else just raises them with a sentence a human can act on.
"""

from __future__ import annotations


class LabError(Exception):
    """Root of everything this project raises on purpose."""


class ConfigError(LabError):
    """Settings are wrong or incomplete in a way we can't guess around."""


class StageError(LabError):
    """A pipeline stage couldn't finish. The message goes straight to the UI,
    so write it like you're telling someone what to do next."""


class PipelineBusy(LabError):
    """A run is already going. One machine, one run."""
