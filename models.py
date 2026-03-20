"""Pydantic v2 request/response models for TruthLens."""

import re
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


_ALLOWED_URL_SCHEMES = {"http", "https"}
_MAX_TEXT_LEN = 5000
_MAX_URL_LEN = 2048
# Control characters (except common whitespace) + null bytes
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class VerifyRequest(BaseModel):
    """Incoming fact-check request — at least one field must be non-empty."""

    text: str | None = None
    url: str | None = None
    image_base64: str | None = None

    @field_validator("text", mode="before")
    @classmethod
    def sanitise_text(cls, v: object) -> str | None:
        if v is None:
            return None
        raw = str(v)
        raw = _CONTROL_CHAR_RE.sub("", raw)
        if len(raw) > _MAX_TEXT_LEN:
            raise ValueError(
                f"text must not exceed {_MAX_TEXT_LEN} characters"
            )
        return raw or None

    @field_validator("url", mode="before")
    @classmethod
    def sanitise_url(cls, v: object) -> str | None:
        if v is None:
            return None
        raw = str(v).strip()
        raw = _CONTROL_CHAR_RE.sub("", raw)
        if len(raw) > _MAX_URL_LEN:
            raise ValueError(
                f"url must not exceed {_MAX_URL_LEN} characters"
            )
        scheme = raw.split("://")[0].lower() if "://" in raw else ""
        if scheme and scheme not in _ALLOWED_URL_SCHEMES:
            raise ValueError(
                f"url scheme '{scheme}' is not allowed; use http or https"
            )
        return raw or None

    @field_validator("image_base64", mode="before")
    @classmethod
    def sanitise_image_base64(cls, v: object) -> str | None:
        if v is None:
            return None
        raw = str(v).strip()
        if not raw:
            return None
        # Validate that it is valid base64
        import base64

        try:
            base64.b64decode(raw, validate=True)
        except Exception:
            raise ValueError("image_base64 is not valid base64")
        return raw

    @model_validator(mode="after")
    def require_at_least_one(self) -> "VerifyRequest":
        if not any([self.text, self.url, self.image_base64]):
            raise ValueError(
                "At least one of text, url, or image_base64 must be provided"
            )
        return self


class VerifyResponse(BaseModel):
    """Structured fact-check verdict returned by POST /verify."""

    verdict: Literal["REAL", "FAKE", "UNVERIFIED"]
    confidence: float
    summary: str
    red_flags: list[str]
    supporting_evidence: list[str]
    sources: list[str]
    searched_queries: list[str]
    disclaimer: str


UNVERIFIED_FALLBACK = VerifyResponse(
    verdict="UNVERIFIED",
    confidence=0.0,
    summary="Unable to parse analysis. Please try again.",
    red_flags=[],
    supporting_evidence=[],
    sources=[],
    searched_queries=[],
    disclaimer=(
        "AI-assisted analysis only. Always verify with authoritative sources."
    ),
)
