"""TruthLens — FastAPI application entry point.

All business logic is delegated to services/. This file contains only:
- App lifecycle (lifespan context manager)
- Route definitions
- Error handling wrappers
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import google.cloud.logging
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models import VerifyRequest, VerifyResponse
from services import gemini as gemini_service

# Attach Cloud Logging to root logger so all ERROR logs go to Cloud Logging
_log_client = google.cloud.logging.Client()
_log_client.setup_logging()

logger = logging.getLogger(__name__)

# Shared async HTTP client — created once at startup, closed at shutdown
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage app-level resources across the application lifetime."""
    global _http_client
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        max_redirects=3,
        follow_redirects=True,
    )
    logger.info("httpx client initialised")
    yield
    await _http_client.aclose()
    logger.info("httpx client closed")


app = FastAPI(
    title="TruthLens",
    description="Gemini-powered fake news detector",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe — returns current model name."""
    return {"status": "ok", "model": gemini_service._MODEL}


@app.get("/trending")
async def get_trending_rumours() -> dict:
    from services.community import get_trending
    return {"trending": await get_trending(limit=5)}


@app.post("/verify", response_model=VerifyResponse)
async def verify(request: VerifyRequest) -> VerifyResponse:
    """Fact-check a claim, URL, or image using Gemini with Google Search grounding."""
    if _http_client is None:
        logger.error("http_client not initialised — lifespan may not have run")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal error. Please try again."},
        )  # type: ignore[return-value]

    try:
        result = await gemini_service.fact_check(request, _http_client)
        import hashlib
        from services.community import record_check
        _hash = hashlib.sha256(
            ((request.text or "") + (request.url or "")).encode()
        ).hexdigest()[:16]
        await record_check(_hash, result.verdict, result.summary, result.harm_severity)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unhandled exception in /verify: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal error. Please try again."},
        )  # type: ignore[return-value]


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: never expose raw exception messages or stack traces to clients."""
    logger.error("Unhandled exception: %s %s — %s", request.method, request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error. Please try again."},
    )
