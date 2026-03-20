"""Gemini service — client initialisation and fact-check logic."""

import base64
import json
import logging
import os
from datetime import date

import httpx
from google import genai
from google.genai import types

from models import UNVERIFIED_FALLBACK, VerifyRequest, VerifyResponse
from services.scraper import fetch_article

logger = logging.getLogger(__name__)

# GCP_PROJECT is required at runtime; defaults to "test-project" only for test imports.
# Cloud Run always injects the real value via --set-env-vars.
GCP_PROJECT: str = os.environ.get("GCP_PROJECT", "test-project")
# GCP_REGION env var is set by deploy.sh; "global" required for gemini-3-flash-preview
_LOCATION: str = os.environ.get("GCP_REGION", "global")
_MODEL: str = "gemini-3-flash-preview"

# Client created once at module level — never per request.
# Falls back to us-central1 + gemini-2.0-flash-001 if location is not "global".
client: genai.Client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT,
    location=_LOCATION,
)

_SYSTEM_PROMPT: str = (
    "You are a fact-checking assistant specialising in identifying misinformation, "
    "propaganda, and fake news. You analyse news claims, articles, and images with "
    "rigorous scepticism. You are especially alert to:\n"
    "- Emotionally manipulative or panic-inducing language\n"
    "- Missing or fake attribution (e.g. \"Government announces...\" with no source)\n"
    "- Implausible statistics or round numbers\n"
    "- Known Indian misinformation patterns: LPG/fuel shortage rumours, "
    "fake government notices, communal tension narratives, WhatsApp forwards\n"
    "- AI-generated image artefacts\n"
    "- Claims contradicting well-established facts\n\n"
    "Today's date is {current_date}. Your knowledge cutoff is January 2025. "
    "For any claim involving events after January 2025, you MUST use the "
    "google_search tool to verify against current sources before rendering a verdict. "
    "When formulating search queries, include the current year for time-sensitive queries.\n\n"
    "Always return ONLY valid JSON matching the requested schema. No markdown, no preamble."
)

_USER_PROMPT_TEMPLATE: str = (
    "Analyse the following content for misinformation.\n\n"
    "{text_section}"
    "{url_section}"
    "\nReturn ONLY this JSON:\n"
    '{{\n'
    '  "verdict": "REAL | FAKE | UNVERIFIED",\n'
    '  "confidence": <float 0.0-1.0>,\n'
    '  "summary": "<one sentence verdict explanation>",\n'
    '  "red_flags": ["<specific red flag, empty array if REAL>"],\n'
    '  "supporting_evidence": ["<supporting fact or signal>"],\n'
    '  "disclaimer": "AI-assisted analysis only. Always verify with authoritative sources."\n'
    '}}\n\n'
    "Rules:\n"
    "- REAL: credible, consistent with known facts, no manipulation signals\n"
    "- FAKE: contains verifiable falsehoods, manipulation tactics, or fabricated content\n"
    "- UNVERIFIED: plausible but cannot be confirmed or denied from available evidence\n"
    "- confidence = your certainty in the verdict, not the truthfulness of the claim\n"
    "- red_flags must be [] for a REAL verdict\n"
    "- Use google_search to verify specific claims before deciding"
)


def _detect_mime_type(image_base64: str) -> str:
    """Detect image MIME type from base64 prefix bytes."""
    try:
        header = base64.b64decode(image_base64[:16])
        if header.startswith(b"\x89PNG"):
            return "image/png"
        if header.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if header.startswith(b"GIF"):
            return "image/gif"
    except Exception:
        pass
    return "image/jpeg"


def _extract_grounding(candidate: object) -> tuple[list[str], list[str]]:
    """Extract sources and searched queries from grounding metadata safely."""
    sources: list[str] = []
    searched_queries: list[str] = []
    try:
        meta = candidate.grounding_metadata  # type: ignore[attr-defined]
        if meta is None:
            return sources, searched_queries
        chunks = getattr(meta, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web:
                uri = getattr(web, "uri", None)
                if uri:
                    sources.append(uri)
        queries = getattr(meta, "web_search_queries", None) or []
        for q in queries:
            query_text = getattr(q, "query", None)
            if query_text:
                searched_queries.append(query_text)
    except Exception as exc:
        logger.error("Failed to extract grounding metadata: %s", exc)
    return sources, searched_queries


def _parse_gemini_json(raw: str) -> dict:  # type: ignore[type-arg]
    """Strip markdown fences and parse JSON from Gemini response."""
    text = raw.strip()
    # Remove triple-backtick fences
    text = text.removeprefix("```json").removeprefix("```")
    text = text.removesuffix("```").strip()
    return json.loads(text)


async def fact_check(
    request: VerifyRequest,
    http_client: httpx.AsyncClient,
) -> VerifyResponse:
    """Run the full fact-check pipeline and return a structured verdict."""
    current_date = date.today().isoformat()
    system_prompt = _SYSTEM_PROMPT.format(current_date=current_date)

    text_section = ""
    url_section = ""

    if request.text:
        text_section = f"CLAIM / TEXT:\n{request.text}\n\n"

    if request.url:
        try:
            article_text = await fetch_article(request.url, http_client)
            url_section = f"ARTICLE CONTENT (from {request.url}):\n{article_text}\n\n"
        except Exception as exc:
            # Scraping failed (403 from NDTV, Cloudflare, X.com, paywalls, etc.).
            # Graceful fallback: pass the raw URL to Gemini and let its Google Search
            # grounding retrieve and verify the content directly. This is more robust
            # than returning a hard 422 error.
            logger.warning(
                "Scraping failed for url=%s (%s) — falling back to Gemini Google Search",
                request.url, exc,
            )
            url_section = (
                f"URL TO ANALYSE: {request.url}\n"
                f"NOTE: Direct article fetch was not possible. "
                f"Use google_search to look up this URL and verify its content.\n\n"
            )

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        text_section=text_section,
        url_section=url_section,
    )

    # Build contents list — text prompt first, then optional image
    contents: list[types.Part] = [types.Part.from_text(text=user_prompt)]

    if request.image_base64:
        # Decode once; pass bytes directly — no re-encoding
        image_bytes = base64.b64decode(request.image_base64)
        mime = _detect_mime_type(request.image_base64)
        contents.append(
            types.Part.from_bytes(data=image_bytes, mime_type=mime)
        )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        # NOTE: response_mime_type="application/json" is intentionally omitted —
        # it conflicts with the google_search grounding tool in the current SDK.
        # JSON is enforced via the prompt and post-processed below.
        thinking_config=types.ThinkingConfig(thinking_budget=512),
    )

    try:
        response = client.models.generate_content(
            model=_MODEL,
            contents=contents,
            config=config,
        )
    except Exception as exc:
        logger.error("Gemini generate_content failed: %s", exc, exc_info=True)
        return UNVERIFIED_FALLBACK

    sources, searched_queries = _extract_grounding(response.candidates[0])

    try:
        parsed = _parse_gemini_json(response.text)
        verdict_str = str(parsed.get("verdict", "UNVERIFIED")).upper()
        if verdict_str not in {"REAL", "FAKE", "UNVERIFIED"}:
            verdict_str = "UNVERIFIED"

        return VerifyResponse(
            verdict=verdict_str,  # type: ignore[arg-type]
            confidence=float(parsed.get("confidence", 0.0)),
            summary=str(parsed.get("summary", "")),
            red_flags=list(parsed.get("red_flags", [])),
            supporting_evidence=list(parsed.get("supporting_evidence", [])),
            sources=sources,
            searched_queries=searched_queries,
            disclaimer=str(
                parsed.get(
                    "disclaimer",
                    "AI-assisted analysis only. Always verify with authoritative sources.",
                )
            ),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.error("Failed to parse Gemini response JSON: %s", exc, exc_info=True)
        fallback = UNVERIFIED_FALLBACK.model_copy(
            update={"sources": sources, "searched_queries": searched_queries}
        )
        return fallback
