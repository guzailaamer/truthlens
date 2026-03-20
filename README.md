# TruthLens 🔍

> AI-powered fake news detector for WhatsApp forwards, viral news, and Instagram screenshots.  
> Built with **Gemini 3 Flash + Google Search grounding** and deployed on **Cloud Run** in minutes.

[![Deploy](https://img.shields.io/badge/Cloud%20Run-Deployed-blue?logo=google-cloud)](https://cloud.google.com/run)

---

## What It Does

TruthLens analyses any of the following and returns a **REAL / FAKE / UNVERIFIED** verdict with reasoning:

| Input | How analysed |
|---|---|
| Pasted text / WhatsApp forward | Sent directly to Gemini with Google Search grounding |
| News article URL | Fetched server-side (httpx + BeautifulSoup), text extracted, sent to Gemini |
| Screenshot (WhatsApp, Instagram, tweet) | Base64-decoded, sent as image bytes to Gemini Vision |

The verdict card includes:
- **Verdict** — REAL, FAKE, or UNVERIFIED (icon + colour + text for accessibility)
- **Confidence score** — displayed as a progress bar
- **Summary** — one-sentence explanation
- **Red flags** — specific suspicious signals found
- **Supporting evidence** — facts supporting the verdict
- **Sources** — URLs Gemini searched via Google grounding
- **Searched queries** — live web searches performed

---

## Architecture

```
Browser (index.html — Tailwind CDN, vanilla JS)
    │
    │  POST /verify  (JSON)
    ▼
FastAPI (main.py)
    │
    ├─── models.py         Pydantic v2 validation + sanitisation
    ├─── services/
    │    ├── scraper.py    async httpx fetch → BeautifulSoup extraction
    │    └── gemini.py     Gemini 3 Flash + Google Search grounding
    │
    └─► Vertex AI (google-genai SDK)
            └── gemini-3-flash-preview @ location=global
                    └── google_search tool (live web grounding)

Cloud Logging ◄─ all ERROR-level events
Cloud Run     ◄─ container (max 3 instances, 512 Mi, concurrency 80)
Cloud Build   ◄─ image build + push to GCR
```

---

## Google Services Used

| Service | How TruthLens uses it |
|---|---|
| **Vertex AI (google-genai SDK)** | Runs `gemini-3-flash-preview` with `thinking_budget=512` to keep latency under 3s. The `google_search` native tool is attached so Gemini grounds every verdict in live web results — no external search APIs needed. |
| **Cloud Run** | Hosts the FastAPI app as a stateless container with `--max-instances 3` (hard budget cap ≤ $5), `--concurrency 80`, `--memory 512Mi`. Auto-scales to zero between requests. |
| **Cloud Logging** | `google.cloud.logging.Client().setup_logging()` is called at startup, attaching Cloud Logging to Python's root logger. Every exception path logs with `severity=ERROR`, giving structured entries in Cloud Console for prod debugging. |
| **Cloud Build** | `deploy.sh` uses `gcloud builds submit` to build the Docker image in the cloud and push it to Google Container Registry — no local Docker daemon required. |

---

## Local Development

```bash
# 1. Install dependencies
pip install -r requirements-dev.txt

# 2. Set required environment variables
export GCP_PROJECT=your-project-id
export GCP_REGION=global          # for gemini-3-flash-preview

# 3. Run the server
uvicorn main:app --reload --port 8080

# 4. Open http://localhost:8080 in your browser
```

---

## Deploy to Cloud Run

```bash
chmod +x deploy.sh
./deploy.sh YOUR_PROJECT_ID us-central1
```

The script:
1. Enables all required GCP APIs
2. Grants Vertex AI IAM role to the compute service account
3. Builds & pushes the Docker image via Cloud Build
4. Deploys to Cloud Run with the correct flags

---

## Run Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

10 test cases — zero real API calls (all Gemini and httpx calls are mocked).

---

## Model Fallback

`gemini-3-flash-preview` requires `location="global"` in the Vertex AI SDK.  
If the preview model is unavailable, update `services/gemini.py`:

```python
_LOCATION = "us-central1"
_MODEL = "gemini-2.0-flash-001"
```

And update `deploy.sh` env var: `GCP_REGION=us-central1`.

---

## Security Notes

- Zero hardcoded secrets — all config via `GCP_PROJECT` / `GCP_REGION` env vars
- Input sanitised before reaching Gemini: null bytes stripped, max lengths enforced, URL scheme validated
- Responses >1 MB rejected to prevent memory exhaustion
- Non-root Docker user (uid 1000)
- Raw exception messages never returned to clients — only `{"detail": "Internal error. Please try again."}`

---

## Submission Checklist

- [x] `GET /health` → `{"status": "ok", "model": "gemini-3-flash-preview"}`
- [x] Verdict card renders for text, URL, and image inputs
- [x] All 10 tests pass with `pytest tests/ -v`
- [x] No secrets in code (`git log --all -p | grep -i "key\|secret\|token"`)
- [x] Cloud Run URL is HTTPS
- [x] README has Google Services Used section
- [ ] GitHub repo set to **public** before submission
