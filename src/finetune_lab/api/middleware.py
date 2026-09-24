"""Request tracing middleware.

Stamps every response with X-Request-ID and X-Response-Time-Ms.

The try/except lives here rather than in a FastAPI exception handler on
purpose. Exception handlers bypass middleware, so if you want the request id
on crash responses too, you have to catch inside the middleware itself.
"""

from __future__ import annotations

import re
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from ..core.logging import log

# A caller-supplied id is echoed into logs and headers, so it only passes if
# it is short and plain. Anything else gets a fresh id of our own.
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


async def request_context(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID", "")
    req_id = incoming if _SAFE_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # we really do want everything here
        log.exception("unhandled error req_id=%s path=%s", req_id, request.url.path)
        response = JSONResponse(
            status_code=500,
            content={
                "detail": "Something broke on our side. The request id below is in the server log.",
                "request_id": req_id,
            },
        )
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response
