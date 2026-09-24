"""App factory.

Run it with:  uvicorn finetune_lab.api.app:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import RedirectResponse

from .. import __author__, __version__
from ..core.config import get_settings
from ..core.logging import setup_logging
from .middleware import request_context
from .routes import chat, health, pipeline, providers

setup_logging()

DESCRIPTION = (
    f"Seven-stage QLoRA fine-tuning pipeline with a live ops console. Built by {__author__}."
)


def create_app() -> FastAPI:
    settings = get_settings()  # fail fast on bad config, create dirs
    app = FastAPI(title="llm-finetune-lab", description=DESCRIPTION, version=__version__)

    # Order matters here, and it is the opposite of what it looks like.
    # Starlette runs the LAST middleware added as the outermost one. CORS has
    # to be outermost, because our tracing middleware answers crashes with its
    # own response, and a 500 that never passes back through CORS reaches the
    # browser with no Access-Control headers. The fetch then fails opaquely
    # and the user sees nothing instead of the error we carefully wrote.
    # So: tracing first, CORS second.
    app.middleware("http")(request_context)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        # Without this the browser hides our own headers from JS, which makes
        # the request id useless to anyone debugging from devtools.
        expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
    )
    # Outermost of all, so a request for a foreign Host is refused before
    # anything else runs.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    for r in (health.router, providers.router, chat.router, pipeline.router):
        app.include_router(r)

    # People open the backend's address in a browser first. Send them to the
    # interactive API docs rather than a bare 404.
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    return app


app = create_app()
