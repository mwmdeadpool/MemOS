"""
Shared-key auth middleware for MemOS REST API.

Reads MEMOS_SHARED_KEY from environment. When set, every request must
present a matching Authorization header. Accepts "Token <key>", "Bearer <key>",
or a raw key. The /health, /docs, /openapi.json, and /redoc paths bypass
auth so monitoring + dashboard still work.

If MEMOS_SHARED_KEY is unset, this middleware is a no-op (preserves
existing OSS behavior for setups not opting in).

Added in mwmdeadpool fork after red-team flagged unauthenticated /product/*
endpoints (High 1, MemOS deployment red-team review 2026-05-20).
"""

import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


BYPASS_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}


def _extract_key(header_value: str) -> str:
    if not header_value:
        return ""
    parts = header_value.split(None, 1)
    if len(parts) == 2 and parts[0].lower() in {"token", "bearer"}:
        return parts[1].strip()
    return header_value.strip()


class SharedKeyAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.shared_key = os.getenv("MEMOS_SHARED_KEY", "")
        self.enabled = bool(self.shared_key)

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        if request.url.path in BYPASS_PATHS:
            return await call_next(request)

        presented = _extract_key(request.headers.get("authorization", ""))
        if not presented or not hmac.compare_digest(presented, self.shared_key):
            return JSONResponse(
                status_code=401,
                content={"code": 401, "message": "Unauthorized", "data": None},
            )

        return await call_next(request)
