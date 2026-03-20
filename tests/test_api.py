"""TruthLens API test suite — 10 test cases, all Gemini/httpx calls mocked.

Run with: pytest tests/ -v
"""

import base64
import json
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_mock_response

VERDICTS = {"REAL", "FAKE", "UNVERIFIED"}

# ---------------------------------------------------------------------------
# Helper: valid minimal base64 PNG (1x1 white pixel)
# ---------------------------------------------------------------------------

_TINY_PNG_B64 = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
).decode()


# ---------------------------------------------------------------------------
# Test 1: GET /health → 200, body matches expected shape
# ---------------------------------------------------------------------------

def test_health_ok(client: TestClient) -> None:
    """GET /health must return 200 with status=ok and the correct model name."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "gemini-3-flash-preview"


# ---------------------------------------------------------------------------
# Test 2: POST /verify with empty body {} → 422
# ---------------------------------------------------------------------------

def test_verify_empty_body(client: TestClient) -> None:
    """Empty body must be rejected with 422."""
    response = client.post("/verify", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 3: POST /verify with all None fields → 422
# ---------------------------------------------------------------------------

def test_verify_all_none(client: TestClient) -> None:
    """All-None fields must be rejected with 422 (at-least-one validator)."""
    response = client.post("/verify", json={"text": None, "url": None, "image_base64": None})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 4: POST /verify with text only → 200, valid verdict
# ---------------------------------------------------------------------------

def test_verify_text_only(client: TestClient, mock_gemini: MagicMock) -> None:
    """Text-only request must return 200 with a valid verdict."""
    mock_resp = make_mock_response(verdict="FAKE", confidence=0.9)
    mock_gemini.aio.models.generate_content.return_value = mock_resp
    response = client.post(
        "/verify",
        json={"text": "LPG cylinders will cost Rs 2000 from Monday, government confirms."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] in VERDICTS
    assert 0.0 <= body["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Test 5: POST /verify with url only → 200 (mock httpx fetch too)
# ---------------------------------------------------------------------------

def test_verify_url_only(client: TestClient, mock_gemini: MagicMock, mock_httpx: AsyncMock) -> None:
    """URL-only request must return 200; both httpx and Gemini are mocked."""
    mock_resp = make_mock_response(verdict="REAL", confidence=0.8)
    mock_gemini.aio.models.generate_content.return_value = mock_resp

    with patch("services.scraper.fetch_article", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = "Test Article\nLegit news content."
        response = client.post("/verify", json={"url": "https://www.thehindu.com/"})

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] in VERDICTS


# ---------------------------------------------------------------------------
# Test 6: POST /verify with image_base64 only → 200
# ---------------------------------------------------------------------------

def test_verify_image_only(client: TestClient, mock_gemini: MagicMock) -> None:
    """Image-only request must return 200 with a valid verdict."""
    mock_resp = make_mock_response(verdict="UNVERIFIED", confidence=0.5)
    mock_gemini.aio.models.generate_content.return_value = mock_resp
    response = client.post("/verify", json={"image_base64": _TINY_PNG_B64})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] in VERDICTS


# ---------------------------------------------------------------------------
# Test 7: POST /verify — Gemini returns malformed JSON → 200, verdict=UNVERIFIED
# ---------------------------------------------------------------------------

def test_verify_malformed_gemini_json(client: TestClient, mock_gemini: MagicMock) -> None:
    """Malformed Gemini JSON must be caught gracefully, returning UNVERIFIED."""
    mock_resp = make_mock_response(raw_text="this is not json {{{{")
    mock_gemini.aio.models.generate_content.return_value = mock_resp
    response = client.post("/verify", json={"text": "Some claim"})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "UNVERIFIED"


# ---------------------------------------------------------------------------
# Test 8: POST /verify with url that returns 404 → 422
# ---------------------------------------------------------------------------

def test_verify_url_404(client: TestClient) -> None:
    """A URL returning HTTP 404 must result in a 422 response."""
    with patch("services.scraper.fetch_article", new_callable=AsyncMock) as mock_fetch:
        from fastapi import HTTPException
        mock_fetch.side_effect = HTTPException(status_code=422, detail="Unable to fetch URL: HTTP 404")
        response = client.post("/verify", json={"url": "https://example.com/not-found"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 9: POST /verify with invalid URL scheme (file://) → 422
# ---------------------------------------------------------------------------

def test_verify_invalid_url_scheme(client: TestClient) -> None:
    """file:// scheme must be rejected at validation time with 422."""
    response = client.post("/verify", json={"url": "file:///etc/passwd"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 10: POST /verify with text exceeding 5000 chars → 422
# ---------------------------------------------------------------------------

def test_verify_text_too_long(client: TestClient) -> None:
    """Text longer than 5000 characters must be rejected with 422."""
    long_text = "A" * 5001
    response = client.post("/verify", json={"text": long_text})
    assert response.status_code == 422
