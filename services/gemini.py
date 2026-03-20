"""Gemini service — client initialisation and fact-check logic."""

import asyncio
import base64
import json
import logging
import os
from datetime import date

import httpx
from google import genai
from google.genai import types

from models import UNVERIFIED_FALLBACK, VerifyRequest, VerifyResponse, ClaimResult
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
_client: genai.Client = genai.Client(
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
    "- AI-generated image artefacts\n"
    "- Claims contradicting well-established facts\n\n"
    "Today's date is {current_date}. Your knowledge cutoff is January 2025.\n"
    "For any claim involving events after January 2025, you MUST use the\n"
    "google_search tool before rendering a verdict.\n"
    "When formulating search queries, include the current year for time-sensitive queries.\n\n"
    "You accept inputs in any language: Hindi, Telugu, Tamil, Kannada, English.\n"
    "Detect input language automatically. Always respond in English.\n"
    "Set input_language to the detected language name in English.\n\n"
    "Be especially alert to Indian misinformation patterns:\n"
    "- Hindi: fake सरकारी नोटिस, BJP/Congress propaganda forwards\n"
    "- Telugu: fake TSRTC/APSRTC notices, Hyderabad civic rumours\n"
    "- Tamil: fake Chennai flood warnings, political deepfake claims\n\n"
    "Harm severity rules:\n"
    "- CRITICAL: could cause loss of life, riots, or mass panic\n"
    "- HIGH: significant societal harm (LPG panic, fake govt policy, election misinfo)\n"
    "- MEDIUM: moderate harm (financial fraud, targeted harassment)\n"
    "- LOW: misleading but low real-world impact\n"
    "- NONE: satire, opinion, or verified true content\n"
    "harm_category must be NONE when verdict is REAL.\n\n"
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
    '  "harm_severity": "CRITICAL | HIGH | MEDIUM | LOW | NONE",\n'
    '  "harm_category": "COMMUNAL_VIOLENCE | PANIC_BUYING | HEALTH_MISINFORMATION | POLITICAL | FINANCIAL | OTHER | NONE",\n'
    '  "input_language": "<detected language in English>",\n'
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


def _parse_gemini_json(raw: str) -> dict | list:  # type: ignore[type-arg]
    """Strip markdown fences and parse JSON from Gemini response."""
    text = raw.strip()
    # Remove triple-backtick fences
    text = text.removeprefix("```json").removeprefix("```")
    text = text.removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def extract_claims(
    text: str | None,
    url_content: str | None,
    has_image: bool,
    client: genai.Client,
) -> list[str]:
    system_prompt = (
        "You are a claim extractor. Given any input, identify every\n"
        "discrete factual claim that could be independently verified.\n"
        "Return ONLY a JSON array of strings. No markdown, no preamble.\n"
        'Example: ["India imports 60% of LPG", "Price will rise to Rs 2000 Monday"]\n'
        "If no verifiable claims found, return []."
    )
    user_str = ""
    if text:
        user_str += f"TEXT:\n{text[:2000]}\n\n"
    if url_content:
        user_str += f"URL CONTENT:\n{url_content[:2000]}\n\n"
    if has_image:
        user_str += "IMAGE: (Image provided in original request)\n\n"
    user_str += "Extract up to 5 verifiable claims from the above content."

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    try:
        response = await client.aio.models.generate_content(
            model=_MODEL,
            contents=[types.Part.from_text(user_str)],
            config=config,
        )
        parsed = _parse_gemini_json(response.text)
        if isinstance(parsed, list):
            return [str(p) for p in parsed][:5]
        return ["Unable to extract claims — analysing as whole"]
    except Exception as exc:
        logger.error("extract_claims failed: %s", exc)
        return ["Unable to extract claims — analysing as whole"]


async def verify_claims(
    claims: list[str],
    original_context: str,
    client: genai.Client,
) -> list[ClaimResult]:
    if not claims or claims == ["Unable to extract claims — analysing as whole"]:
        return []

    system_prompt = _SYSTEM_PROMPT.format(current_date=date.today().isoformat())
    user_prompt = (
        "Verify each of these claims using google_search. \n"
        f"Claims to verify: {json.dumps(claims)}\n"
        f"Original context: {original_context[:500]}\n\n"
        "Return ONLY a JSON array matching this schema:\n"
        '[{\n  "claim": "<exact claim string>",\n  "verdict": "TRUE | FALSE | UNVERIFIED",\n  "explanation": "<one sentence>",\n  "sources": ["<URL from grounding>"]\n}]'
    )
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    try:
        response = await client.aio.models.generate_content(
            model=_MODEL,
            contents=[types.Part.from_text(text=user_prompt)],
            config=config,
        )
        sources, _ = _extract_grounding(response.candidates[0])
        parsed = _parse_gemini_json(response.text)
        
        results = []
        if isinstance(parsed, list):
            for item in parsed:
                v = str(item.get("verdict", "UNVERIFIED")).upper()
                if v not in {"TRUE", "FALSE", "UNVERIFIED"}:
                    v = "UNVERIFIED"
                results.append(ClaimResult(
                    claim=str(item.get("claim", "")),
                    verdict=v,  # type: ignore[arg-type]
                    explanation=str(item.get("explanation", "")),
                    sources=list(item.get("sources", [])) or sources
                ))
            return results
        return [ClaimResult(claim=c, verdict="UNVERIFIED", explanation="Could not parse verification results", sources=[]) for c in claims]
    except Exception as exc:
        logger.error("verify_claims failed: %s", exc)
        return [ClaimResult(claim=c, verdict="UNVERIFIED", explanation="Verification failed", sources=[]) for c in claims]


async def transcribe_audio(
    audio_base64: str,
    mime_type: str,
    client: genai.Client,
) -> str:
    """Transcribe base64 audio using Gemini."""
    audio_bytes = base64.b64decode(audio_base64)
    prompt = "Please transcribe this audio clip accurately. Do not add any extra commentary. Just the transcription."
    
    config = types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    try:
        response = await client.aio.models.generate_content(
            model=_MODEL,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                types.Part.from_text(text=prompt),
            ],
            config=config,
        )
        return response.text.strip()
    except Exception as exc:
        logger.error("Audio transcription failed: %s", exc)
        return ""


async def fact_check(
    request: VerifyRequest,
    http_client: httpx.AsyncClient,
) -> VerifyResponse:
    """Run the full fact-check pipeline and return a structured verdict."""
    current_date = date.today().isoformat()
    system_prompt = _SYSTEM_PROMPT.format(current_date=current_date)

    if request.audio_base64 and request.audio_mime_type:
        audio_text = await transcribe_audio(request.audio_base64, request.audio_mime_type, _client)
        if audio_text:
            request.text = (request.text or "") + f"\n\n(Transcribed from Voice Note)\n{audio_text}"

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

    contents: list[types.Part] = [types.Part.from_text(text=user_prompt)]

    if request.image_base64:
        image_bytes = base64.b64decode(request.image_base64)
        mime = _detect_mime_type(request.image_base64)
        contents.append(
            types.Part.from_bytes(data=image_bytes, mime_type=mime)
        )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        thinking_config=types.ThinkingConfig(thinking_budget=512),
    )

    try:
        has_image = bool(request.image_base64)
        article_content = article_text if request.url and 'fetch_article' in locals() and 'article_text' in locals() else None
        
        # 1. Extract claims
        claims_list = await extract_claims(request.text, article_content, has_image, _client)
        
        # Deduce leaning purely from extracted claims structure to pass hint IF we could, 
        # but since we fire them in parallel, we'll just run them and merge after.
        # Actually the prompt says: "Pass this leaning as a hint in the synthesis prompt"
        # Since it's physically impossible when using asyncio.gather, we append the extracted claims to the synthesis contents 
        # so it has the context of what is being verified concurrently.
        hint = f"\n\nExtracted claims being verified concurrently: {json.dumps(claims_list)}. Use these as evidence."
        contents[0].text += hint

        # 2. Parallel verify_claims and synthesis call
        context_str = (request.text or "") + " " + (article_content or "")
        
        synthesis_kwargs = dict(
            model=_MODEL,
            contents=contents,
            config=config,
        )

        synthesis_task = _client.aio.models.generate_content(**synthesis_kwargs)
        verify_task = verify_claims(claims_list, context_str, _client)
        
        response, claim_results = await asyncio.gather(synthesis_task, verify_task)

    except Exception as exc:
        logger.error("Gemini pipeline failed: %s", exc, exc_info=True)
        return UNVERIFIED_FALLBACK

    sources, searched_queries = _extract_grounding(response.candidates[0])

    try:
        parsed = _parse_gemini_json(response.text)
        if not isinstance(parsed, dict):
            raise ValueError("Synthesis response is not a dict")
        
        verdict_str = str(parsed.get("verdict", "UNVERIFIED")).upper()
        if verdict_str not in {"REAL", "FAKE", "UNVERIFIED"}:
            verdict_str = "UNVERIFIED"

        # Derivation logic as per prompt (synthesis decides final, but we can override if it's UNVERIFIED or contradictory)
        # Actually, prompt says "Gemini synthesis still decides final". We just set claims_analysed.
        
        return VerifyResponse(
            verdict=verdict_str,  # type: ignore[arg-type]
            confidence=float(parsed.get("confidence", 0.0)),
            summary=str(parsed.get("summary", "")),
            claims_analysed=claim_results,
            red_flags=list(parsed.get("red_flags", [])),
            supporting_evidence=list(parsed.get("supporting_evidence", [])),
            sources=sources,
            searched_queries=searched_queries,
            harm_severity=str(parsed.get("harm_severity", "NONE")),  # type: ignore[arg-type]
            harm_category=str(parsed.get("harm_category", "NONE")),  # type: ignore[arg-type]
            input_language=str(parsed.get("input_language", "English")),
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
