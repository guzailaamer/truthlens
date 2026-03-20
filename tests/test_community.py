"""Tests for community firestore stub."""
import pytest
from unittest.mock import patch, AsyncMock
from services.community import record_check, get_trending

@pytest.mark.asyncio
async def test_record_check_no_db():
    # If _db is None, this should return silently
    with patch("services.community._db", None):
        await record_check("hash123", "REAL", "Summary", "LOW")
        # Should not raise any exceptions

@pytest.mark.asyncio
async def test_get_trending_no_db():
    with patch("services.community._db", None):
        res = await get_trending(5)
        assert res == []

def test_community_importability():
    import services.community
    assert hasattr(services.community, "record_check")
