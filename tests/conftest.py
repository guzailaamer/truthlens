"""Shared pytest fixtures for TruthLens test suite."""

import os
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Patch GCP_PROJECT env var and both GCP clients globally before any app imports.
# This prevents module-level genai.Client() and google.cloud.logging.Client() from
# attempting real GCP connections when the test environment has no credentials.
os.environ.setdefault("GCP_PROJECT", "test-project")
os.environ.setdefault("GCP_REGION", "global")

with patch("google.cloud.logging.Client"), \
     patch("google.genai.Client"):
    import main  # noqa: E402 — must import AFTER env/patch setup


# ---------------------------------------------------------------------------
# App client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient — GCP clients are mocked at import time."""
    with TestClient(main.app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Gemini response mock builders
# ---------------------------------------------------------------------------

def make_mock_response(
    verdict: str = "UNVERIFIED",
    confidence: float = 0.75,
    summary: str = "Test summary.",
    red_flags: list | None = None,
    supporting_evidence: list | None = None,
    sources: list | None = None,
    searched_queries: list | None = None,
    raw_text: str | None = None,
) -> MagicMock:
    """Build a mock Gemini response object with grounding metadata."""
    import json

    red_flags = red_flags or []
    supporting_evidence = supporting_evidence or []
    sources = sources or []
    searched_queries = searched_queries or []

    payload = {
        "verdict": verdict,
        "confidence": confidence,
        "summary": summary,
        "red_flags": red_flags,
        "supporting_evidence": supporting_evidence,
        "disclaimer": "AI-assisted analysis only. Always verify with authoritative sources.",
    }

    response = MagicMock()
    response.text = raw_text if raw_text is not None else json.dumps(payload)

    # Build mock grounding_metadata
    grounding_meta = MagicMock()
    chunks = []
    for src in sources:
        chunk = MagicMock()
        chunk.web.uri = src
        chunks.append(chunk)
    grounding_meta.grounding_chunks = chunks

    queries = []
    for q_text in searched_queries:
        q = MagicMock()
        q.query = q_text
        queries.append(q)
    grounding_meta.web_search_queries = queries

    candidate = MagicMock()
    candidate.grounding_metadata = grounding_meta
    response.candidates = [candidate]

    return response
