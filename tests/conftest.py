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

@pytest.fixture
def mock_gemini(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock the Gemini client inside services.gemini."""
    from unittest.mock import AsyncMock
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock()
    monkeypatch.setattr("services.gemini._client", mock_client)
    return mock_client


@pytest.fixture
def mock_httpx(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Mock the httpx client and fetch_article to avoid real network calls."""
    from unittest.mock import AsyncMock
    mock_client = AsyncMock()
    monkeypatch.setattr("main._http_client", mock_client)
    # Also patch fetch_article just in case it's called directly
    patcher = patch("services.scraper.fetch_article", new_callable=AsyncMock)
    mock_fetch = patcher.start()
    mock_fetch.return_value = "<html><body>Legit article text</body></html>"
    # To stop patched correctly, we must add a finalizer but for simplicity we let monkeypatch handle what we can.
    # We will use monkeypatch for fetch_article as well:
    mock_article = AsyncMock(return_value="<html><body>Legit article text</body></html>")
    monkeypatch.setattr("services.scraper.fetch_article", mock_article)
    return mock_article


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
    harm_severity: str = "NONE",
    harm_category: str = "NONE",
    input_language: str = "English",
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
        "harm_severity": harm_severity,
        "harm_category": harm_category,
        "input_language": input_language,
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
