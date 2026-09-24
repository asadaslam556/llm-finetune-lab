"""HTTP layer. Thin on purpose: routes validate, call in, and shape output."""

from .app import create_app

__all__ = ["create_app"]
