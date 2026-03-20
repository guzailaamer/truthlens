# TruthLens — Antigravity Project Guide
> 45-minute warmup sprint · Gemini 3 Flash + Google Search grounding · Cloud Run

---

## The one-line pitch
Paste a WhatsApp forward, a news URL, or a screenshot → get a **REAL / FAKE / UNVERIFIED** verdict in 3 seconds, grounded in live web sources. Grok for WhatsApp forwards.

---

## Master prompt — paste this into Antigravity first

```
You are building a production-ready app called "TruthLens" for a 45-minute
Google PromptWars hackathon warmup. Move fast — prioritise working code over
perfect code, but do not skip types, Pydantic models, or error handling.

=== PROBLEM STATEMENT ===
Build a Gemini-powered fake news detector that takes unstructured inputs
(text claim, news URL, or image screenshot) and returns a structured
verdict card: REAL / FAKE / UNVERIFIED with reasoning and confidence.
The app targets WhatsApp forwards, Instagram screenshots, and viral rumours —
especially Indian regional misinformation (LPG shortage, fuel panic, communal tension).

=== STACK ===
- Python 3.11 + FastAPI
- Gemini 3 Flash (gemini-3-flash-preview) via google-genai SDK — NEVER use Pro or 2.0
- Google Search grounding — native tool, no Tavily, no external search APIs
- httpx + BeautifulSoup for URL article extraction (server-side, no headless browser)
- Single Cloud Run deployment, --max-instances 3 (hard budget cap, $5 total)
- Cloud Logging for all errors
- NO database, NO Cloud Storage, NO auth middleware — not needed for warmup

=== MODEL CONFIG ===
SDK: google-genai>=1.51.0 (NOT google-cloud-aiplatform)
Model string: gemini-3-flash-preview
Location: global  ← REQUIRED for Gemini 3, not us-central1
thinking_budget: 512  ← LOW thinking, keeps latency acceptable

from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=contents,
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        thinking_config=types.ThinkingConfig(thinking_budget=512),
        response_mime_type="application/json",
    )
)

Extract grounding_metadata from response.candidates[0].grounding_metadata
to populate sources[] and searched_queries[] in the response.

=== FILE STRUCTURE ===
truthlens/
├── main.py
├── models.py
├── services/
│   ├── gemini.py
│   └── scraper.py
├── static/
│   └── index.html
├── tests/
│   └── test_api.py
├── Dockerfile
├── deploy.sh
├── requirements.txt
└── README.md

=== ENDPOINTS ===
GET  /          → serve static/index.html
GET  /health    → {"status": "ok", "model": "gemini-3-flash-preview"}
POST /verify    → main fact-check endpoint

=== POST /verify ===
Request (Pydantic v2, at least one field required):
{
  "text":         "optional — claim, headline, or pasted forward",
  "url":          "optional — news article URL",
  "image_base64": "optional — base64 encoded image (jpg/png)"
}

Validation: if all three fields are None/empty, return 422.

Response:
{
  "verdict":           "REAL | FAKE | UNVERIFIED",
  "confidence":        0.0-1.0,
  "summary":           "one sentence explaining the verdict",
  "red_flags":         ["specific suspicious element found"],
  "supporting_evidence": ["fact or signal supporting the verdict"],
  "sources":           ["URL from grounding_metadata"],
  "searched_queries":  ["queries Gemini searched"],
  "disclaimer":        "AI-assisted analysis only. Always verify with authoritative sources."
}

=== URL HANDLING (scraper.py) ===
- async httpx fetch, timeout=10s
- BeautifulSoup html.parser — strip <script>, <style>, <nav>, <footer>, <header>
- Extract: <title>, <meta name="description">, <article> or <main> or <body> text
- Truncate to 3000 chars max before passing to Gemini
- On fetch failure: raise HTTPException(422, detail="Unable to fetch URL: {reason}")

=== IMAGE HANDLING ===
- Accept image_base64 string from request
- Pass to Gemini as: types.Part.from_bytes(data=base64.b64decode(image_base64), mime_type="image/jpeg")
- Do not validate or transform — pass as-is

=== GEMINI PROMPT TEMPLATES (gemini.py) ===

SYSTEM_PROMPT = """
You are a fact-checking assistant specialising in identifying misinformation,
propaganda, and fake news. You analyse news claims, articles, and images with
rigorous scepticism. You are especially alert to:
- Emotionally manipulative or panic-inducing language
- Missing or fake attribution (e.g. "Government announces..." with no source)
- Implausible statistics or round numbers
- Known Indian misinformation patterns: LPG/fuel shortage rumours,
  fake government notices, communal tension narratives, WhatsApp forwards
- AI-generated image artefacts
- Claims contradicting well-established facts

Today's date is {current_date}. Your knowledge cutoff is January 2025.
For any claim involving events after January 2025, you MUST use the
google_search tool to verify against current sources before rendering a verdict.
When formulating search queries, include the current year for time-sensitive queries.

Always return ONLY valid JSON matching the requested schema. No markdown, no preamble.
"""

USER_PROMPT = """
Analyse the following content for misinformation.

{text_section}
{url_section}

Return ONLY this JSON:
{{
  "verdict": "REAL | FAKE | UNVERIFIED",
  "confidence": <float 0.0-1.0>,
  "summary": "<one sentence verdict explanation>",
  "red_flags": ["<specific red flag, empty array if REAL>"],
  "supporting_evidence": ["<supporting fact or signal>"],
  "disclaimer": "AI-assisted analysis only. Always verify with authoritative sources."
}}

Rules:
- REAL: credible, consistent with known facts, no manipulation signals
- FAKE: contains verifiable falsehoods, manipulation tactics, or fabricated content
- UNVERIFIED: plausible but cannot be confirmed or denied from available evidence
- confidence = your certainty in the verdict, not the truthfulness of the claim
- red_flags must be [] for a REAL verdict
- Use google_search to verify specific claims before deciding
"""

Build contents list dynamically:
- Always add the text USER_PROMPT as first Part
- If url content extracted: append as additional text Part
- If image_base64 provided: append as bytes Part (inline_data)

After response, extract:
  meta = response.candidates[0].grounding_metadata
  sources = [chunk.web.uri for chunk in meta.grounding_chunks if chunk.web]
  searched_queries = [q.query for q in meta.web_search_queries]

Parse response.text with json.loads — ALWAYS strip markdown fences first:
  clean = response.text.strip().removeprefix("```json").removesuffix("```").strip()

Wrap json.loads in try/except — on failure return:
  {"verdict": "UNVERIFIED", "confidence": 0.0,
   "summary": "Unable to parse analysis. Please try again.",
   "red_flags": [], "supporting_evidence": [],
   "sources": [], "searched_queries": [], "disclaimer": "..."}

=== FRONTEND (static/index.html) ===
Single HTML file, vanilla JS, Tailwind CDN. No React, no build step.

Layout (top to bottom):
1. Header bar: "TruthLens" (bold) + "Fact-check anything in seconds" (muted)
   Subtext: "For WhatsApp forwards, viral news, Instagram screenshots"

2. Input card (white card, subtle border):
   - Textarea (4 rows): placeholder "Paste a claim, WhatsApp forward, or headline..."
   - Text input: placeholder "Or paste a news article URL..."
   - File input (styled as drag zone): "Upload a screenshot (WhatsApp, Instagram, tweet...)"
     On file select: read as base64, store in JS variable, show filename confirmation
   - "Analyse" button — full width, disabled + greyed if all inputs empty
     Loading state: spinner + "Analysing with Gemini..." while in-flight

3. Result card (hidden until response, animate in):
   - Verdict pill (large, centered):
     REAL     → green background (#16a34a), white text
     FAKE     → red background (#dc2626), white text
     UNVERIFIED → amber background (#d97706), white text
   - Confidence bar: thin (4px) progress bar below pill, colour matches verdict
   - Confidence label: "87% confident" in small muted text
   - Summary text: 16px, normal weight, centered, max-width 600px
   - "Why?" section (always expanded):
     If red_flags non-empty: red-tinted box, bullet list
     supporting_evidence: green-tinted box, bullet list
   - "Sources searched" (collapsible, default collapsed):
     searched_queries as grey pills
     sources as clickable links (open in new tab)
   - Disclaimer: small, muted, italic, bottom of card

4. Error state: red inline alert below button, shows error message

JS behaviour:
- On file upload: FileReader.readAsDataURL → strip data:...;base64, prefix → store
- On submit: build JSON body with only non-empty fields, POST to /verify
- Disable button + show spinner during fetch
- On 422: show "Please provide at least one input"
- On 5xx: show "Analysis failed. Please try again."

=== CODE QUALITY ===
- Type annotations on all functions, no bare Any
- async def on all route handlers
- try/except on every external call (Gemini, httpx) — never expose raw exceptions
- Cloud Logging: from google.cloud import logging — log ERROR on every exception
- Dockerfile: python:3.11-slim, non-root user (uid 1000), no dev deps

=== DOCKERFILE ===
FROM python:3.11-slim
WORKDIR /app
RUN useradd -u 1000 -m appuser
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chown -R appuser /app
USER appuser
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

=== REQUIREMENTS.TXT ===
fastapi
uvicorn[standard]
google-genai>=1.51.0
httpx
beautifulsoup4
pydantic>=2.0
google-cloud-logging

=== DEPLOY.SH ===
#!/bin/bash
set -e
PROJECT_ID=$1
REGION=${2:-us-central1}

echo "Setting project..."
gcloud config set project $PROJECT_ID

echo "Enabling APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  aiplatform.googleapis.com logging.googleapis.com

echo "Granting Vertex AI access to default compute SA..."
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/aiplatform.user"

echo "Building and pushing image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/truthlens

echo "Deploying to Cloud Run..."
gcloud run deploy truthlens \
  --image gcr.io/$PROJECT_ID/truthlens \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars GCP_PROJECT=$PROJECT_ID,GCP_REGION=global \
  --max-instances 3 \
  --memory 512Mi \
  --timeout 60

echo "Done. Service URL:"
gcloud run services describe truthlens --region $REGION --format='value(status.url)'

=== TESTS (tests/test_api.py) — 3 tests only ===
1. GET /health → 200, body contains "gemini-3-flash-preview"
2. POST /verify with {} → 422
3. POST /verify with text="LPG shortage in Hyderabad confirmed, cylinders to cost Rs 2000 from Monday"
   → 200, verdict in ["REAL","FAKE","UNVERIFIED"]
   Mock google.genai.Client with unittest.mock.patch — return a fake response object
   with .text = valid JSON string and .candidates[0].grounding_metadata = mock object

=== START INSTRUCTIONS ===
1. Show me the complete file structure only — no code yet
2. Wait for my approval (I will say "go")
3. Then generate ALL files in one pass without stopping
4. After all files are generated, run deploy.sh automatically
5. Show me the Cloud Run URL when done
```

---

## 45-minute clock

| Time | Action |
|---|---|
| 0:00 | Paste master prompt into Antigravity, approve file structure |
| 0:03 | Say "go" — Antigravity generates all files |
| 0:10 | Check `gemini.py` — grounding_metadata extraction is the most likely failure point |
| 0:15 | Deploy starts |
| 0:28 | Cloud Run URL live — hit `/health` |
| 0:32 | Test with: *"LPG cylinders will cost ₹2000 from Monday, govt confirms"* |
| 0:38 | Push to GitHub — `git init`, `git push`, set repo public |
| 0:42 | Submit Cloud Run URL + GitHub URL on Hack2skill |
| 0:45 | Done |

---

## Steering prompts — keep these ready to paste

**If grounding_metadata extraction crashes:**
```
In gemini.py, wrap grounding_metadata extraction in try/except.
If grounding_metadata is None or missing attributes, set sources=[] and
searched_queries=[] silently — do not raise an exception.
```

**If Gemini returns markdown-wrapped JSON:**
```
In gemini.py after getting response.text, add:
  text = response.text.strip()
  if text.startswith("```"):
      text = text.split("```")[1]
      if text.startswith("json"):
          text = text[4:]
  result = json.loads(text.strip())
```

**If deploy fails on Vertex AI permission:**
```
Add to deploy.sh before the build step:
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

**If Gemini 3 Flash preview is unavailable in the region:**
```
Change client init to:
client = genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")
And model to: "gemini-2.0-flash-001" as fallback — same SDK, same tool support.
```

**If frontend file input base64 is broken:**
```
In index.html, fix the FileReader:
  reader.onload = (e) => {
    imageBase64 = e.target.result.split(',')[1];
    document.getElementById('file-label').textContent = file.name;
  };
```

---

## Demo script (test these three inputs after deploy)

```
# Test 1 — Classic Indian WhatsApp rumour (should return FAKE)
text: "URGENT: LPG cylinders will cost ₹2000 from Monday. Government has issued
official notice. Share with family before prices increase. Jai Hind 🇮🇳"

# Test 2 — Real news URL (should return REAL or UNVERIFIED)
url: https://www.thehindu.com/  ← use any real recent news URL

# Test 3 — Geopolitical misinformation (should return FAKE or UNVERIFIED)
text: "Israel has officially surrendered to Hamas. UN confirms ceasefire signed
this morning. CNN and BBC hiding the news. Share before it gets deleted."
```

---

## Submission checklist

- [ ] `GET /health` returns `{"status": "ok", "model": "gemini-3-flash-preview"}`
- [ ] Verdict card renders for all three input types
- [ ] GitHub repo is **public** — verify in incognito before submitting
- [ ] No secrets in git: `git log --all -p | grep -i "key\|secret\|token"`
- [ ] Cloud Run URL is HTTPS (always is — just confirm it loads)
- [ ] README has: what it does, architecture, deploy instructions, GCP services used
