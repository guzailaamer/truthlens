"""Tests for claim decomposition and verification."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from services.gemini import extract_claims, verify_claims
from models import ClaimResult

@pytest.mark.asyncio
async def test_extract_claims_happy_path():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '["Claim 1", "Claim 2"]'
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)
    
    claims = await extract_claims("Some text", None, False, mock_client)
    assert claims == ["Claim 1", "Claim 2"]

@pytest.mark.asyncio
async def test_extract_claims_malformed_json():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = 'NOT JSON AT ALL'
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)
    
    claims = await extract_claims("Some text", "url_content", False, mock_client)
    assert claims == ["Unable to extract claims — analysing as whole"]

@pytest.mark.asyncio
async def test_verify_claims():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '[{"claim": "Claim 1", "verdict": "TRUE", "explanation": "x", "sources": ["a.com"]}]'
    
    candidate = MagicMock()
    meta = MagicMock()
    meta.grounding_chunks = []
    meta.web_search_queries = []
    candidate.grounding_metadata = meta
    mock_resp.candidates = [candidate]
    
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)
    
    results = await verify_claims(["Claim 1"], "Context", mock_client)
    assert len(results) == 1
    assert isinstance(results[0], ClaimResult)
    assert results[0].claim == "Claim 1"
    assert results[0].verdict == "TRUE"
