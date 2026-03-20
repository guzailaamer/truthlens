"""URL scraper — async httpx fetch + BeautifulSoup extraction."""

import logging

import httpx
from bs4 import BeautifulSoup
from fastapi import HTTPException

logger = logging.getLogger(__name__)

_MAX_CONTENT_BYTES = 1_048_576  # 1 MB
_TRUNCATE_CHARS = 3000
_FETCH_TIMEOUT = 10.0
_MAX_REDIRECTS = 3
_STRIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript"}


async def fetch_article(url: str, client: httpx.AsyncClient) -> str:
    """Fetch a URL and extract readable article text, truncated to 3000 chars."""
    try:
        response = await client.get(
            url,
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
        )
    except httpx.TimeoutException as exc:
        logger.error("URL fetch timed out: url=%s error=%s", url, exc)
        raise HTTPException(
            status_code=422,
            detail=f"Unable to fetch URL: request timed out after {_FETCH_TIMEOUT}s",
        ) from exc
    except httpx.TooManyRedirects as exc:
        logger.error("Too many redirects: url=%s error=%s", url, exc)
        raise HTTPException(
            status_code=422,
            detail="Unable to fetch URL: too many redirects",
        ) from exc
    except httpx.RequestError as exc:
        logger.error("HTTP request error: url=%s error=%s", url, exc)
        raise HTTPException(
            status_code=422,
            detail=f"Unable to fetch URL: {type(exc).__name__}",
        ) from exc

    if response.status_code != 200:
        logger.error("Non-200 response: url=%s status=%d", url, response.status_code)
        raise HTTPException(
            status_code=422,
            detail=f"Unable to fetch URL: HTTP {response.status_code}",
        )

    content_length = int(response.headers.get("content-length", 0))
    if content_length > _MAX_CONTENT_BYTES:
        raise HTTPException(
            status_code=422,
            detail="Unable to fetch URL: response exceeds 1 MB size limit",
        )

    html = response.text
    return _extract_text(html, url)


def _extract_text(html: str, url: str) -> str:
    """Parse HTML with BeautifulSoup and extract clean article text."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and isinstance(meta_tag, object):
        meta_desc = meta_tag.get("content", "")  # type: ignore[union-attr]

    body_text = ""
    for selector in ("article", "main", "body"):
        container = soup.find(selector)
        if container:
            body_text = container.get_text(separator=" ", strip=True)
            break

    combined = "\n".join(filter(None, [title, meta_desc, body_text]))
    return combined[:_TRUNCATE_CHARS]
